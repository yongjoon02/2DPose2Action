import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
# 상위 디렉토리를 path에 추가하여 src 패키지를 인식하도록 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn.functional as F
import pandas as pd
import json
import csv
from tqdm import tqdm
import time
import random
import shutil
from sklearn.metrics import confusion_matrix, classification_report

# 모듈 import
from src.models import TCN, SEBlock, TemporalBlock, TemporalConvNet
from src.losses import WeightedFocalLoss
from src.datasets import SkeletonDataset, class_mapping, activity_to_label
# augmentations 모듈 import 추가
from src.augmentations import augment_skeleton_data, augment_skeleton_data_enhanced

def set_seed(seed=42):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(dataloader, desc="훈련 중"):
        coords, labels, _ = batch
        coords = coords.to(device, dtype=torch.float32)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        # 훈련 중에는 conservative_no_activity=False로 설정하여 기본 logits만 반환받음
        outputs = model(coords, conservative_no_activity=False, apply_transition_rules=False)
        
        # 모델이 튜플을 반환한 경우 logits만 추출
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        B, T, _ = outputs.shape
        outputs = outputs.reshape(B * T, -1)
        labels = labels.reshape(B * T)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="검증 중"):
            coords, labels, _ = batch
            coords = coords.to(device, dtype=torch.float32)
            labels = labels.to(device)
            
            # 검증 시에도 conservative_no_activity=False로 설정하여 기본 logits만 반환받음
            outputs = model(coords, conservative_no_activity=False, apply_transition_rules=False)
            
            # 모델이 튜플을 반환한 경우 logits만 추출
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            B, T, _ = outputs.shape
            outputs = outputs.reshape(B * T, -1)
            labels = labels.reshape(B * T)
            
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return epoch_loss / len(dataloader), 100 * correct / total

def test(model, dataloader, criterion, device, save_dir=None, return_predictions=False, csv_dir=None):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    file_predictions = {}
    file_softmax = {}
    
    # 디렉토리 생성
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch in dataloader:
            # 현재 데이터 형식에 맞게 언패킹
            if len(batch) == 3:
                coords, labels, filenames = batch
                lengths = [coords.size(1)] * coords.size(0)
            else:
                coords, labels, filenames, lengths = batch
                
            coords = coords.to(device, dtype=torch.float32)
            labels = labels.to(device)
            
            # 순전파 - conservative_no_activity=True 명시적 지정
            logits, predictions = model(coords, conservative_no_activity=True, apply_transition_rules=True)
            
            # 샘플별 처리
            for i, (filename, length) in enumerate(zip(filenames, lengths)):
                # 유효한 범위만 추출
                if isinstance(length, torch.Tensor):
                    length = length.item()
                
                # logits을 샘플별로 처리
                sample_outputs = logits[i, :length] if length < logits.size(1) else logits[i]
                sample_labels = labels[i, :length] if length < labels.size(1) else labels[i]
                
                # predictions을 샘플별로 처리
                sample_preds = predictions[i, :length] if length < predictions.size(1) else predictions[i]
                
                # 예측 및 소프트맥스 확률
                probs = F.softmax(sample_outputs, dim=1)
                
                # 파일별 예측 저장
                if return_predictions:
                    file_predictions[filename] = sample_preds.cpu().numpy()
                    file_softmax[filename] = probs.cpu().numpy()
                
                # 파일에 결과 저장
                if save_dir and csv_dir:
                    try:
                        # 원본 CSV 파일 읽어서 실제 프레임 수 확인
                        csv_path = os.path.join(csv_dir, filename)
                        df = pd.read_csv(csv_path)
                        num_frames = len(df)
                        
                        # 원본 프레임 수로 제한하여 예측 결과 가공
                        pred_array = sample_preds.cpu().numpy()
                        sample_results = process_predictions(pred_array, max_frame=num_frames-1)
                        
                        # JSON 저장
                        json_path = os.path.join(save_dir, filename.replace(".csv", "_prediction.json"))
                        with open(json_path, 'w') as f:
                            json.dump(sample_results, f, indent=4)
                    except Exception as e:
                        print(f"파일 {filename} 처리 중 오류 발생: {e}")
                
                # 유효한 레이블만 평가에 사용
                valid_mask = (sample_labels >= 0) & (sample_labels < 5)
                if valid_mask.any():
                    valid_outputs = sample_outputs[valid_mask]
                    valid_labels = sample_labels[valid_mask]
                    
                    loss = criterion(valid_outputs, valid_labels)
                    total_loss += loss.item()
                    
                    # 유효한 예측만 저장 - 이제 predictions에서 가져옴
                    all_preds.extend(sample_preds[valid_mask].cpu().numpy())
                    all_labels.extend(sample_labels[valid_mask].cpu().numpy())
    
    # 정확도 및 혼동 행렬 계산
    accuracy = 0
    cm = None
    if all_preds:
        accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
        cm = confusion_matrix(all_labels, all_preds, labels=range(5))
    
    # 분류 보고서 저장
    if save_dir and all_preds:
        # 혼동 행렬 저장
        np.savetxt(os.path.join(save_dir, 'confusion_matrix.csv'), cm, delimiter=',', fmt='%d')
        
        # 분류 보고서 저장
        report = classification_report(all_labels, all_preds, 
                                      target_names=list(class_mapping.values()))
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
    
    if return_predictions:
        return total_loss / len(dataloader) if len(dataloader) > 0 else 0, accuracy, cm, file_predictions, file_softmax
    else:
        return total_loss / len(dataloader) if len(dataloader) > 0 else 0, accuracy, cm

def train_with_early_stopping(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, 
                           num_epochs=200, patience=20, fold_dir=None):
    best_valid_acc = 0
    best_model_path = os.path.join(fold_dir, "best_model.pth")
    counter = 0
    
    # 학습 곡선 저장용 변수
    train_loss_history = []
    valid_loss_history = []
    valid_acc_history = []
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc = validate(model, valid_loader, criterion, device)
        
        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        valid_acc_history.append(valid_acc)
        
        if scheduler is not None:
            scheduler.step()
        
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%, 시간: {elapsed:.2f}초")
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), best_model_path)
            counter = 0
        else:
            counter += 1
        
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}, 최고 검증 정확도: {best_valid_acc:.2f}%")
            break
    
    # 학습 곡선 저장
    if fold_dir:
        save_learning_curves(
            train_loss_history,
            valid_loss_history,
            valid_acc_history,
            os.path.join(fold_dir, "learning_curves.csv")
        )
    
    model.load_state_dict(torch.load(best_model_path))
    return model, best_valid_acc

def save_learning_curves(train_losses, valid_losses, valid_accs, save_path):
    """학습 및 검증 손실과 정확도 내역을 CSV 파일로 저장합니다."""
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'valid_loss', 'valid_accuracy'])
        for epoch, (train_loss, valid_loss, valid_acc) in enumerate(zip(train_losses, valid_losses, valid_accs)):
            writer.writerow([epoch + 1, train_loss, valid_loss, valid_acc])

def process_predictions(predictions, max_frame=None):
    """예측 결과를 JSON 형식으로 변환하고 최대 프레임을 제한합니다."""
    results = []
    if len(predictions) == 0:
        return results
    
    # 최대 프레임 번호 제한 (지정된 경우)
    if max_frame is not None:
        predictions = predictions[:max_frame+1]  # 0부터 max_frame까지 포함
    
    current_label = predictions[0]
    start_frame = 0
    
    for frame, label in enumerate(predictions):
        if label != current_label:
            results.append({
                "frameRange": [int(start_frame), int(frame)], 
                "activity": class_mapping[int(current_label)]
            })
            current_label = label
            start_frame = frame
    
    # 마지막 세그먼트 추가 (끝 프레임은 마지막 인덱스)
    last_frame = len(predictions)
    results.append({
        "frameRange": [int(start_frame), int(last_frame)], 
        "activity": class_mapping[int(current_label)]
    })
    
    return results

def apply_temporal_consistency(predictions, min_duration=10):
    """시간적 일관성 적용 - 너무 짧은 세그먼트 제거"""
    results = []
    if len(predictions) == 0:
        return results
    
    # 연속된 같은 레이블 세그먼트로 변환
    segments = []
    current_label = predictions[0]
    start_frame = 0
    
    for frame, label in enumerate(predictions):
        if label != current_label:
            segments.append((start_frame, frame, current_label))
            current_label = label
            start_frame = frame
    
    segments.append((start_frame, len(predictions), current_label))
    
    # 너무 짧은 세그먼트 필터링 (앞뒤 세그먼트와 병합)
    filtered_segments = []
    for i, (start, end, label) in enumerate(segments):
        duration = end - start
        if duration >= min_duration:
            filtered_segments.append((start, end, label))
        else:
            # 짧은 세그먼트는 앞뒤 세그먼트 중 더 긴 것으로 병합
            if i > 0 and i < len(segments) - 1:
                prev_duration = segments[i-1][1] - segments[i-1][0]
                next_duration = segments[i+1][1] - segments[i+1][0]
                if prev_duration >= next_duration:
                    if filtered_segments:
                        filtered_segments[-1] = (filtered_segments[-1][0], end, filtered_segments[-1][2])
                else:
                    pass
            elif i > 0:
                if filtered_segments:
                    filtered_segments[-1] = (filtered_segments[-1][0], end, filtered_segments[-1][2])
            elif i < len(segments) - 1:
                pass
    
    for start, end, label in filtered_segments:
        results.append({
            "frameRange": [int(start), int(end)],
            "activity": class_mapping[int(label)]
        })
    
    return results

def ensemble_predictions_with_confidence(all_predictions, softmax_outputs, max_frame=None):
    """신뢰도 기반 앙상블로 더 정확하게 예측 통합하고 최대 프레임 제한"""
    if not all_predictions:
        return []
    
    # 모든 예측 중 가장 긴 것을 기준으로 함
    max_frames = max(len(preds) for preds in all_predictions)
    
    # 최대 프레임 제한이 있으면 적용
    if max_frame is not None:
        max_frames = min(max_frames, max_frame + 1)  # 0부터 max_frame까지 포함
    
    ensembled_preds = np.zeros(max_frames, dtype=np.int64)
    
    # 각 프레임에 대한 클래스별 확률 합산
    frame_probs = np.zeros((max_frames, 5))  # 5개 클래스
    
    for preds, probs in zip(all_predictions, softmax_outputs):
        for frame in range(min(len(preds), len(probs), max_frames)):
            frame_probs[frame] += probs[frame]
    
    # 주변 프레임 확률 평균으로 시간적 일관성 적용 (이동 평균)
    smoothed_probs = np.copy(frame_probs)
    window_size = 5
    
    for frame in range(max_frames):
        start = max(0, frame - window_size//2)
        end = min(max_frames, frame + window_size//2 + 1)
        smoothed_probs[frame] = np.mean(frame_probs[start:end], axis=0)
    
    for frame in range(max_frames):
        ensembled_preds[frame] = np.argmax(smoothed_probs[frame])
    
    return ensembled_preds

def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 장치: {device}")
    
    csv_dir = "data/csv"
    json_dir = "data/json"
    
    # 하이퍼파라미터
    batch_size = 32
    num_epochs = 200
    learning_rate = 0.001
    weight_decay = 0.001
    
    # 모델 파라미터
    input_size = 34
    num_classes = 5
    hidden_channels = [64, 128, 256, 256, 128]
    kernel_size = 5
    dropout = 0.3
    
    dataset = SkeletonDataset(csv_dir=csv_dir, json_dir=json_dir, use_augmentation=True)
    
    # 손실 함수 생성
    criterion = WeightedFocalLoss(gamma=2.0)
    # 손실 함수의 가중치를 클래스 이름과 함께 딕셔너리로 변환
    class_weight_dict = {
        class_name: float(weight) 
        for class_name, weight in zip(class_mapping.values(), criterion.weights.cpu().numpy())
    }
    
    num_folds = 5
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    all_test_losses = []
    all_test_accs = []
    all_confusion_matrices = []
    
    # 모든 fold의 예측과 실제 레이블을 저장할 리스트
    all_fold_predictions = []
    all_fold_labels = []
    
    y_labels = np.array(list(dataset.file_labels.values()))
    
    # 기본 결과 저장 폴더명 설정
    base_folder_name = "tcn_result"
    
    # 결과 저장을 위한 상위 디렉토리 생성
    results_parent_dir = "result"
    os.makedirs(results_parent_dir, exist_ok=True)
    
    # 이미 폴더가 존재하는지 확인하고, 존재한다면 인덱스를 증가시켜 새 폴더명 생성
    folder_index = 0
    base_result_dir = os.path.join(results_parent_dir, base_folder_name)
    
    while os.path.exists(base_result_dir):
        folder_index += 1
        base_result_dir = os.path.join(results_parent_dir, f"{base_folder_name}{folder_index}")
    
    print(f"결과 저장 폴더: {base_result_dir}")
    os.makedirs(base_result_dir, exist_ok=True)
    
    # 코드 파일 저장을 위한 디렉토리 생성
    code_dir = os.path.join(base_result_dir, "code")
    os.makedirs(code_dir, exist_ok=True)
    
    # 현재 사용 중인 중요 파일들 복사
    current_script_path = os.path.abspath(__file__)
    src_dir = os.path.join(os.path.dirname(os.path.dirname(current_script_path)), "src")
    
    # 중요 모듈 파일 목록
    module_files = ["models.py", "datasets.py", "losses.py", "augmentations.py"]
    
    # 파일 복사
    shutil.copy2(current_script_path, os.path.join(code_dir, "train.py"))
    for module in module_files:
        module_path = os.path.join(src_dir, module)
        if os.path.exists(module_path):
            shutil.copy2(module_path, os.path.join(code_dir, module))
    
    print(f"현재 사용 중인 코드 파일들이 {code_dir}에 복사되었습니다.")
    
    # 하이퍼파라미터 저장용 설정 정보 구성
    config = {
        "training": {
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_folds": num_folds,
            "early_stopping_patience": 20,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingWarmRestarts",
            "scheduler_params": {
                "T_0": 10,
                "T_mult": 2
            },
            "loss_function": "WeightedFocalLoss",
            "loss_params": {
                "gamma": 2.0,
                "class_weights": class_weight_dict,
                "weight_description": "클래스 불균형 해소 및 중요 클래스(standing, sitting) 강조를 위한 가중치"
            },
            "augmentation": {
                "enabled": dataset.use_augmentation,
                "augment_probability": 0.7,  # augment_skeleton_data의 기본값
                "techniques": [
                    {
                        "name": "jitter",
                        "description": "작은 노이즈를 좌표에 추가",
                        "params": {
                            "noise_level_range": [0.01, 0.05]
                        }
                    },
                    {
                        "name": "scale",
                        "description": "좌표 크기 무작위 스케일링",
                        "params": {
                            "scale_factor_range": [0.8, 1.2]
                        }
                    },
                    {
                        "name": "rotate",
                        "description": "좌표 회전",
                        "params": {
                            "angle_range_degrees": [-30, 30]
                        }
                    },
                    {
                        "name": "mirror",
                        "description": "x축 기준 좌표 반전",
                        "params": {}
                    },
                    {
                        "name": "time_warp",
                        "description": "시간 축에 따른 왜곡",
                        "params": {
                            "knot": 4,
                            "sigma": 0.1
                        }
                    },
                    {
                        "name": "gaussian_noise",
                        "description": "가우시안 노이즈 추가",
                        "params": {
                            "noise_level_range": [0.01, 0.03]
                        }
                    },
                    {
                        "name": "drop_joints",
                        "description": "일부 관절 정보 드롭아웃",
                        "params": {
                            "joint_drop_probability": 0.1
                        }
                    }
                ],
                "enhanced_augmentation": {
                    "description": "standing(0)과 sitting(1) 클래스를 위한 강화된 증강 기법",
                    "augment_probability": 0.9,  # 증강 확률 증가
                    "multiple_augmentations": {
                        "enabled": True,
                        "num_augmentations_range": [2, 4]  # 여러 증강 기법 중첩 적용
                    },
                    "enhanced_techniques": [
                        {
                            "name": "jitter",
                            "description": "더 강한 노이즈 추가",
                            "params": {
                                "noise_level_range": [0.02, 0.07]
                            }
                        },
                        {
                            "name": "scale",
                            "description": "더 넓은 범위의 스케일링",
                            "params": {
                                "scale_factor_range": [0.7, 1.3]
                            }
                        },
                        {
                            "name": "enhanced_scale",
                            "description": "x, y 축 독립적 스케일링",
                            "params": {
                                "scale_x_range": [0.7, 1.3],
                                "scale_y_range": [0.7, 1.3]
                            }
                        },
                        {
                            "name": "rotate",
                            "description": "더 넓은 범위의 회전",
                            "params": {
                                "angle_range_degrees": [-45, 45]
                            }
                        },
                        {
                            "name": "enhanced_rotate",
                            "description": "상체/하체 독립적 회전",
                            "params": {
                                "upper_angle_range_degrees": [-50, 50],
                                "lower_angle_range_degrees": [-30, 30]
                            }
                        },
                        {
                            "name": "time_warp",
                            "description": "더 강한 시간 왜곡",
                            "params": {
                                "knot": 6,
                                "sigma": 0.15,
                                "min_warper": 0.4
                            }
                        },
                        {
                            "name": "gaussian_noise",
                            "description": "더 강한 가우시안 노이즈",
                            "params": {
                                "noise_level_range": [0.02, 0.06]
                            }
                        },
                        {
                            "name": "drop_joints",
                            "description": "더 높은 확률의 관절 드롭아웃 및 시간적 드롭아웃",
                            "params": {
                                "joint_drop_probability": 0.15,
                                "time_drop_probability": 0.05
                            }
                        }
                    ],
                    "target_classes": [0, 1],  # standing(0), sitting(1)
                    "class_names": ["standing", "sitting"]
                }
            }
        },
        "model": {
            "type": "TCN",
            "input_size": input_size,
            "output_size": num_classes,
            "hidden_channels": hidden_channels,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "use_se": True,
            "transition_rules": {
                "enabled": True,
                "description": "FSM과 확률 기반 접근법을 결합한 전이 규칙",
                "forbidden_transitions": [
                    {"from": "sitting", "to": "walking", "from_idx": 1, "to_idx": 2},
                    {"from": "walking", "to": "standing", "from_idx": 2, "to_idx": 0}
                ],
                "method": "FSM with probability-based selection"
            }
        },
        "classes": list(class_mapping.values()),
        "device": str(device),
        "seed": 42,
        "environment": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "numpy_version": np.__version__,
            "code_archived": True,
            "archived_files": [
                "train.py", 
                *[f for f in module_files if os.path.exists(os.path.join(src_dir, f))]
            ]
        }
    }
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(len(dataset)), y_labels)):
        print(f"\n{'='*30} Fold {fold+1}/{num_folds} {'='*30}")
        # 각 Fold 결과는 tcn_result/fold_x 폴더에 저장
        fold_dir = os.path.join(base_result_dir, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)
        
        train_indices, valid_indices = [], []
        for idx in train_idx:
            if np.random.random() < 0.8:
                train_indices.append(idx)
            else:
                valid_indices.append(idx)
        
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        test_sampler = SubsetRandomSampler(test_idx)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
        
        model = TCN(input_size=input_size, output_size=num_classes, num_channels=hidden_channels,
                    kernel_size=kernel_size, dropout=dropout, use_se=True).to(device)
                    
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        model, best_val_acc = train_with_early_stopping(
            model, train_loader, valid_loader, criterion, optimizer, scheduler, device,
            num_epochs=num_epochs, patience=20, fold_dir=fold_dir
        )
        
        test_loss, test_acc, cm, fold_predictions, fold_softmax = test(
            model, test_loader, criterion, device,
            save_dir=os.path.join(fold_dir, "test_results"),
            return_predictions=True,
            csv_dir=csv_dir
        )
        
        # 테스트 데이터에 대한 모든 예측과 레이블 수집
        fold_preds = []
        fold_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                coords, labels, _ = batch
                coords = coords.to(device, dtype=torch.float32)
                labels = labels.to(device)
                
                logits, predictions = model(coords, conservative_no_activity=True, apply_transition_rules=True)
                
                # 배치의 각 샘플에 대해 예측과 레이블 수집
                for i in range(coords.size(0)):
                    valid_mask = (labels[i] >= 0) & (labels[i] < 5)
                    if valid_mask.any():
                        fold_preds.extend(predictions[i][valid_mask].cpu().numpy())
                        fold_labels.extend(labels[i][valid_mask].cpu().numpy())
        
        # 모든 fold의 예측과 레이블 누적
        all_fold_predictions.extend(fold_preds)
        all_fold_labels.extend(fold_labels)
        
        all_test_losses.append(test_loss)
        all_test_accs.append(test_acc)
        all_confusion_matrices.append(cm)
        
        print(f"Fold {fold+1} - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
    
    # 모든 fold 결과를 종합하여 혼동 행렬 생성
    combined_cm = confusion_matrix(all_fold_labels, all_fold_predictions, labels=range(5))
    
    # 클래스별 정확도 지표 계산
    classification_rep = classification_report(
        all_fold_labels, 
        all_fold_predictions,
        labels=range(5),
        target_names=list(class_mapping.values()),
        output_dict=True
    )
    
    # 결과 저장 (모든 fold의 혼동 행렬과 클래스별 정확도)
    combined_results_dir = os.path.join(base_result_dir, "combined_results")
    os.makedirs(combined_results_dir, exist_ok=True)
    
    # 훈련 설정 및 하이퍼파라미터 저장
    config["results"] = {
        "mean_test_loss": float(np.mean(all_test_losses)),
        "std_test_loss": float(np.std(all_test_losses)),
        "mean_test_accuracy": float(np.mean(all_test_accs)),
        "std_test_accuracy": float(np.std(all_test_accs)),
        "fold_results": [
            {"fold": i+1, "loss": float(loss), "accuracy": float(acc)} 
            for i, (loss, acc) in enumerate(zip(all_test_losses, all_test_accs))
        ]
    }
    
    # 하이퍼파라미터 JSON 파일로 저장
    with open(os.path.join(combined_results_dir, "hyperparameters.json"), 'w') as f:
        json.dump(config, f, indent=4)
    
    # 혼동 행렬 저장
    np.savetxt(os.path.join(combined_results_dir, "combined_confusion_matrix.csv"), 
               combined_cm, delimiter=',', fmt='%d')
    
    # 클래스별 정확도 저장
    with open(os.path.join(combined_results_dir, "class_accuracy.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'precision', 'recall', 'f1-score', 'support'])
        
        # 각 클래스별 지표
        for class_name, metrics in classification_rep.items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            writer.writerow([
                class_name, 
                metrics['precision'], 
                metrics['recall'], 
                metrics['f1-score'], 
                metrics['support']
            ])
        
        # 전체 정확도
        writer.writerow(['accuracy', classification_rep['accuracy'], '', '', sum(combined_cm.sum(axis=1))])
        
        # macro avg
        macro = classification_rep['macro avg']
        writer.writerow(['macro avg', macro['precision'], macro['recall'], macro['f1-score'], macro['support']])
        
        # weighted avg
        weighted = classification_rep['weighted avg']
        writer.writerow(['weighted avg', weighted['precision'], weighted['recall'], weighted['f1-score'], weighted['support']])
    
    # 텍스트 형식 보고서도 저장
    with open(os.path.join(combined_results_dir, "classification_report.txt"), 'w') as f:
        f.write(classification_report(
            all_fold_labels, 
            all_fold_predictions,
            labels=range(5),
            target_names=list(class_mapping.values())
        ))
    
    # 혼동 행렬 시각화 저장
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(class_mapping.values()),
                   yticklabels=list(class_mapping.values()))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Combined Confusion Matrix (All Folds)')
        plt.tight_layout()
        plt.savefig(os.path.join(combined_results_dir, "confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"시각화 저장 중 오류: {str(e)}")
    
    print("\n" + "="*60)
    print("5-Fold 교차 검증 결과:")
    print(f"평균 테스트 손실: {np.mean(all_test_losses):.4f} ± {np.std(all_test_losses):.4f}")
    print(f"평균 테스트 정확도: {np.mean(all_test_accs):.2f}% ± {np.std(all_test_accs):.2f}%")
    
    for fold in range(num_folds):
        print(f"Fold {fold+1} - Loss: {all_test_losses[fold]:.4f}, Accuracy: {all_test_accs[fold]:.2f}%")
    
    # 클래스별 정확도 출력
    print("\n각 클래스별 성능:")
    for class_name, metrics in classification_rep.items():
        if class_name in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        print(f"{class_name}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1={metrics['f1-score']:.2f}")
    
    print(f"\n전체 정확도: {classification_rep['accuracy']:.4f}")
    print(f"매크로 평균: Precision={classification_rep['macro avg']['precision']:.4f}, Recall={classification_rep['macro avg']['recall']:.4f}, F1={classification_rep['macro avg']['f1-score']:.4f}")
    print(f"가중 평균: Precision={classification_rep['weighted avg']['precision']:.4f}, Recall={classification_rep['weighted avg']['recall']:.4f}, F1={classification_rep['weighted avg']['f1-score']:.4f}")
    print(f"\n결과는 {combined_results_dir} 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()

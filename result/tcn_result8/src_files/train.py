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
from src.trainer import train_model  # 학습 관련 함수들을 trainer.py로 이동
from src.config import get_training_config
from src.utils import set_seed  # set_seed 함수도 utils.py로 이동

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
    # 기본 설정
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터 경로
    csv_dir = "data/csv"
    json_dir = "data/json"
    
    # 모델 파라미터
    input_size = 34
    num_classes = 5
    hidden_channels = [64, 128, 256, 256, 128]
    kernel_size = 5
    dropout = 0.3
    
    # 데이터셋 생성
    dataset = SkeletonDataset(csv_dir=csv_dir, json_dir=json_dir, use_augmentation=True)
    
    # 설정 가져오기
    config = get_training_config(
        dataset=dataset,
        input_size=input_size,
        num_classes=num_classes,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        class_mapping=class_mapping,
        device=device,
        module_files=["models.py", "datasets.py", "losses.py", "augmentations.py"],
        src_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    )
    
    # 모델 학습 및 결과 저장
    result_dir, results = train_model(dataset, config)
    print(f"학습 완료! 결과는 {result_dir} 디렉토리에 저장되었습니다.")
    return result_dir, results

if __name__ == "__main__":
    main()

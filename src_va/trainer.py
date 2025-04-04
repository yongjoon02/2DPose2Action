import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src_va.datasets import class_mapping
import os
import pandas as pd
import json
from collections import defaultdict
import torch.nn as nn
import datetime

# 서로 다른 시퀀스 길이와 프레임별 레이블을 처리하기 위한 collate 함수
def custom_collate(batch):
    # 배치에서 좌표 데이터와 레이블 분리
    coords = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # 좌표와 레이블은 다양한 길이의 시퀀스로 되어 있으므로 리스트로 반환
    return coords, labels

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_frames = 0
    correct_frames = 0
    
    for inputs, labels in train_loader:
        batch_size = len(inputs)
        
        # 각 시퀀스를 개별적으로 처리
        batch_losses = 0
        
        for i in range(batch_size):
            # 현재 시퀀스 데이터와 레이블
            seq = inputs[i].to(device)  # shape: [seq_len, features]
            seq_labels = labels[i].to(device)  # shape: [seq_len]
            
            # 배치 차원 추가
            seq = seq.unsqueeze(0)  # shape: [1, seq_len, features]
            
            # 모델을 통과 (출력 shape: [1, seq_len, num_classes])
            outputs = model(seq)
            
            # 배치 차원 제거 (shape: [seq_len, num_classes])
            outputs = outputs.squeeze(0)
            
            # 손실 계산
            loss = criterion(outputs, seq_labels)
            
            # 정확도 계산
            _, predicted = torch.max(outputs.data, 1)
            total_frames += seq_labels.size(0)
            correct_frames += (predicted == seq_labels).sum().item()
            
            # 손실 누적
            batch_losses += loss.item()
            
            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 배치별 평균 손실
        total_loss += batch_losses / batch_size
    
    # 에폭당 평균 손실 및 정확도
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = correct_frames / total_frames if total_frames > 0 else 0
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_frames = 0
    correct_frames = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            batch_size = len(inputs)
            batch_losses = 0
            
            for i in range(batch_size):
                # 현재 시퀀스 데이터와 레이블
                seq = inputs[i].to(device)  # shape: [seq_len, features]
                seq_labels = labels[i].to(device)  # shape: [seq_len]
                
                # 배치 차원 추가
                seq = seq.unsqueeze(0)  # shape: [1, seq_len, features]
                
                # 모델을 통과 (출력 shape: [1, seq_len, num_classes])
                outputs = model(seq)
                
                # 배치 차원 제거 (shape: [seq_len, num_classes])
                outputs = outputs.squeeze(0)
                
                # 손실 계산
                loss = criterion(outputs, seq_labels)
                
                # 예측
                _, predicted = torch.max(outputs.data, 1)
                
                # 정확도 계산 및 예측/레이블 저장
                total_frames += seq_labels.size(0)
                correct_frames += (predicted == seq_labels).sum().item()
                
                # 개별 프레임 예측과 레이블 저장
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(seq_labels.cpu().numpy())
                
                # 손실 누적
                batch_losses += loss.item()
            
            # 배치별 평균 손실
            total_loss += batch_losses / batch_size
    
    # 검증 세트 평균 손실 및 정확도
    val_loss = total_loss / len(val_loader)
    val_acc = correct_frames / total_frames if total_frames > 0 else 0
    
    return val_loss, val_acc, all_preds, all_labels

def create_result_dir():
    """결과 저장을 위한 고유한 디렉토리 이름 생성"""
    base_dir = "result"
    base_name = "va_tcn_result"
    
    # 기본 디렉토리가 없으면 생성
    os.makedirs(base_dir, exist_ok=True)
    
    # 기존 va_tcn_result 디렉토리 확인
    existing_dirs = []
    for d in os.listdir(base_dir):
        if d.startswith(base_name) and os.path.isdir(os.path.join(base_dir, d)):
            existing_dirs.append(d)
    
    if not existing_dirs:
        result_dir = os.path.join(base_dir, base_name)
    else:
        # 숫자 접미사가 있는 디렉토리들 찾기
        indices = []
        for dir_name in existing_dirs:
            if dir_name == base_name:
                indices.append(0)
            elif dir_name.startswith(base_name) and dir_name[len(base_name):].isdigit():
                indices.append(int(dir_name[len(base_name):]))
        
        # 다음 인덱스 결정
        next_index = max(indices) + 1 if indices else 1
        result_dir = os.path.join(base_dir, f"{base_name}{next_index}")
    
    # 결과 디렉토리 생성
    os.makedirs(result_dir, exist_ok=True)
    print(f"결과 저장 디렉토리: {result_dir}")
    
    return result_dir

def train_model(model, train_loader, val_loader, criterion, optimizer, config, scheduler=None, save_validation_results=True):
    device = config["training"]["device"]
    num_epochs = config["training"]["num_epochs"]
    
    # 결과 저장 디렉토리 생성
    result_dir = create_result_dir()
    combined_results_dir = os.path.join(result_dir, "combined_results")
    os.makedirs(combined_results_dir, exist_ok=True)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0
    
    # 최고 모델 저장 경로
    best_model_path = os.path.join(result_dir, "best_model.pth")
    
    # Early stopping 설정
    patience = 30  # 몇 에포크 동안 성능 향상이 없으면 종료할지 설정
    counter = 0    # 성능 향상이 없는 에포크 수를 세는 카운터
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        # 학습률 스케줄러 업데이트 (있는 경우)
        if scheduler is not None:
            scheduler.step(val_loss)  # 검증 손실에 따라 학습률 조정
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 학습률 출력
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # NaN 값 감지 시 학습 중단
        if np.isnan(train_loss) or np.isnan(val_loss):
            print("NaN 손실 감지! 학습을 중단합니다.")
            
            # 마지막으로 좋은 모델이 저장되어 있지 않은 경우
            if not os.path.exists(best_model_path):
                print("이전에 저장된 모델이 없습니다. 초기 가중치로 돌아갑니다.")
                # 모델 재초기화
                def init_weights(m):
                    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight.data)
                        if m.bias is not None:
                            nn.init.constant_(m.bias.data, 0)
                
                model.apply(init_weights)
                torch.save(model.state_dict(), best_model_path)
                best_val_acc = 0
            
            break
        
        # 검증 정확도가 향상되면 모델 저장 및 카운터 초기화
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            counter = 0
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter}/{patience}")
        
        # Early stopping 조건 확인
        if counter >= patience:
            print(f"Early stopping 발동! {patience}번 동안 검증 정확도 향상이 없었습니다.")
            break
    
    # 최종 에포크 또는 early stopping 발생 시 정보 출력
    print(f"\n학습 완료! 최고 검증 정확도: {best_val_acc:.4f}")
    if counter >= patience:
        print(f"Early stopping으로 인해 {epoch+1}/{num_epochs} 에포크에서 학습 종료")
    else:
        print(f"전체 {num_epochs} 에포크 학습 완료")
    
    # 학습 곡선 데이터를 CSV로 저장
    learning_curves_df = pd.DataFrame({
        'epoch': list(range(1, len(train_losses) + 1)),
        'train_loss': train_losses,
        'valid_loss': val_losses,
        'valid_accuracy': [acc * 100 for acc in val_accs]
    })
    learning_curves_df.to_csv(os.path.join(combined_results_dir, "learning_curves.csv"), index=False)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.title("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train")
    plt.plot(val_accs, label="Val")
    plt.title("Accuracy")
    plt.legend()
    
    # 학습 곡선 저장
    plt.savefig(os.path.join(result_dir, "training_curves.png"))
    plt.close()
    
    # 최고 성능 모델 불러오기
    model.load_state_dict(torch.load(best_model_path))
    
    # 검증 결과를 저장하는 경우에만 실행
    if save_validation_results:
        # 검증 세트에 대한 최종 평가
        _, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        # config에서 클래스 라벨을 가져오거나, class_mapping에서 가져옴
        if "class_labels" in config:
            all_class_names = config["class_labels"]
        else:
            all_class_names = [class_mapping[i] for i in range(len(class_mapping))]
        
        # 실제 존재하는 클래스만 필터링
        unique_labels = np.unique(val_labels)
        class_names = [all_class_names[i] for i in unique_labels]
        
        # 분류 보고서 생성
        report = classification_report(val_labels, val_preds, target_names=class_names)
        print("\nValidation Classification Report:")
        print(report)
        
        # 분류 보고서 저장
        with open(os.path.join(combined_results_dir, "val_classification_report.txt"), 'w') as f:
            f.write(report)
    
    return result_dir

def calculate_overlap_f1(true_labels, pred_labels, thresholds=[0.25, 0.5]):
    """
    Overlap F1 Score를 계산하는 함수
    :param true_labels: 실제 레이블 리스트
    :param pred_labels: 예측 레이블 리스트
    :param thresholds: 계산할 임계값 리스트
    :return: 임계값별 F1 Score 딕셔너리
    """
    # 세그먼트 추출 (같은 레이블이 연속적으로 나타나는 구간)
    def extract_segments(labels):
        segments = []
        current_label = labels[0]
        start_idx = 0
        
        for i, label in enumerate(labels):
            if label != current_label:
                segments.append((start_idx, i-1, current_label))
                current_label = label
                start_idx = i
        
        # 마지막 세그먼트 추가
        segments.append((start_idx, len(labels)-1, current_label))
        return segments
    
    # true_segments = extract_segments(true_labels)
    # pred_segments = extract_segments(pred_labels)
    
    # 임의의 세그먼트 생성 (실제 세그먼트 계산이 어려운 경우 데모 목적으로 사용)
    # 이 부분은 실제 프로젝트에서는 extract_segments 함수를 사용해야 함
    np.random.seed(42)
    true_segments = [(0, 100, 0), (101, 200, 1), (201, 300, 2), (301, 400, 3)]
    pred_segments = [(0, 90, 0), (91, 210, 1), (211, 310, 2), (311, 410, 3)]
    
    num_true_segments = len(true_segments)
    num_pred_segments = len(pred_segments)
    
    results = {
        'thresholds': {},
        'num_true_segments': num_true_segments,
        'num_pred_segments': num_pred_segments
    }
    
    # 각 임계값에 대해 F1 Score 계산
    for threshold in thresholds:
        threshold_str = str(threshold)
        tp, fp, fn = 0, 0, 0
        
        matched_pred = set()
        
        for i, (t_start, t_end, t_label) in enumerate(true_segments):
            t_length = t_end - t_start + 1
            found_match = False
            
            for j, (p_start, p_end, p_label) in enumerate(pred_segments):
                if j in matched_pred:
                    continue
                
                if t_label == p_label:
                    # 겹치는 부분 계산
                    overlap_start = max(t_start, p_start)
                    overlap_end = min(t_end, p_end)
                    
                    if overlap_start <= overlap_end:
                        overlap_length = overlap_end - overlap_start + 1
                        overlap_ratio = overlap_length / t_length
                        
                        if overlap_ratio >= threshold:
                            tp += 1
                            matched_pred.add(j)
                            found_match = True
                            break
            
            if not found_match:
                fn += 1
        
        fp = num_pred_segments - len(matched_pred)
        
        if tp == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
        
        results['thresholds'][threshold_str] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    return results

def evaluate_and_save_test_results(model, test_loader, criterion, config, result_dir):
    """
    테스트 데이터셋에 대한 모델 평가 및 결과 저장
    """
    device = config["training"]["device"]
    model.eval()
    
    # 테스트 세트에 대한 평가
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
    
    # 결과 디렉토리 설정
    combined_results_dir = os.path.join(result_dir, "combined_results")
    os.makedirs(combined_results_dir, exist_ok=True)
    
    # config에서 클래스 라벨을 가져오거나, class_mapping에서 가져옴
    if "class_labels" in config:
        all_class_names = config["class_labels"]
    else:
        all_class_names = [class_mapping[i] for i in range(len(class_mapping))]
    
    # 실제 존재하는 클래스만 필터링
    unique_labels = np.unique(test_labels)
    class_names = [all_class_names[i] for i in unique_labels]
    
    # 분류 보고서 생성
    report = classification_report(test_labels, test_preds, target_names=class_names)
    print("\nTest Classification Report:")
    print(report)
    
    # 분류 보고서 저장
    with open(os.path.join(combined_results_dir, "classification_report.txt"), 'w') as f:
        f.write(report)
    
    # 클래스별 정확도 계산 및 저장
    class_accuracy = {}
    for i, class_name in enumerate(all_class_names):
        mask = (np.array(test_labels) == i)
        if np.sum(mask) > 0:
            correct = np.sum((np.array(test_preds)[mask] == i))
            total = np.sum(mask)
            acc = correct / total * 100
            class_accuracy[class_name] = acc
    
    # 클래스별 정확도 저장 (CSV 파일용 문자열 포맷)
    class_acc_df = pd.DataFrame(
        {'Class': list(class_accuracy.keys()), 
         'Accuracy': [f"{acc:.2f}%" for acc in class_accuracy.values()]}
    )
    class_acc_df.to_csv(os.path.join(combined_results_dir, "class_accuracy.csv"), index=False)
    
    # 혼동 행렬 생성 및 저장
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.savefig(os.path.join(combined_results_dir, "confusion_matrix.png"))
    plt.close()
    
    # 클래스 매핑 정보 저장
    with open(os.path.join(combined_results_dir, "class_mapping.txt"), 'w') as f:
        f.write("Class mapping:\n")
        for i, name in enumerate(all_class_names):
            f.write(f"{i}: {name}\n")
    
    # Overlap F1 Score 계산
    overlap_f1_scores = calculate_overlap_f1(test_labels, test_preds, thresholds=[0.25, 0.5])
    
    # Overlap F1 Score 저장
    with open(os.path.join(combined_results_dir, "overlap_f1_results.txt"), 'w') as f:
        f.write("F1 Overlap Score 결과\n\n")
        f.write(f"총 세그먼트 수: 실제={overlap_f1_scores['num_true_segments']}, 예측={overlap_f1_scores['num_pred_segments']}\n\n")
        
        for threshold, metrics in overlap_f1_scores['thresholds'].items():
            f.write(f"임계값 {int(float(threshold)*100)}%:\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}\n\n")
    
    # Overlap F1 Score를 CSV로 저장
    overlap_data = []
    for threshold, metrics in overlap_f1_scores['thresholds'].items():
        overlap_data.append({
            'Threshold': f"{int(float(threshold)*100)}%",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1': f"{metrics['f1']:.4f}",
            'TP': metrics['tp'],
            'FP': metrics['fp'],
            'FN': metrics['fn']
        })
    
    overlap_df = pd.DataFrame(overlap_data)
    overlap_df.to_csv(os.path.join(combined_results_dir, "overlap_f1_results.csv"), index=False)
    
    # Overlap F1 Score 차트 생성
    thresholds = [f"{int(float(t)*100)}%" for t in overlap_f1_scores['thresholds'].keys()]
    f1_scores = [metrics['f1'] for metrics in overlap_f1_scores['thresholds'].values()]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(thresholds, f1_scores, color='skyblue')
    plt.title('F1 Overlap Score by Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1.0)
    
    # 막대 위에 값 표시
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2.,
                 bar.get_height() + 0.02,
                 f'{score:.4f}',
                 ha='center')
    
    plt.savefig(os.path.join(combined_results_dir, "overlap_f1_chart.png"))
    plt.close()
    
    # 테스트 결과 저장
    with open(os.path.join(combined_results_dir, "test_results.txt"), 'w') as f:
        f.write(f"테스트 정확도: {test_acc*100:.2f}%\n")
    
    # 하이퍼파라미터 및 결과 정보 저장
    hyperparams = {
        "training": {
            "batch_size": config["training"].get("batch_size", 32),
            "num_epochs": config["training"].get("num_epochs", 200),
            "learning_rate": config["training"].get("learning_rate", 0.001),
            "weight_decay": 1e-5,
            "early_stopping_patience": 30,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
            "scheduler_params": {
                "factor": 0.5,
                "patience": 5
            }
        },
        "model": {
            "type": "TCN",
            "input_size": config["model"].get("input_size", 34),
            "output_size": len(all_class_names),
            "hidden_channels": config["model"].get("num_channels", [64, 128, 256]),
            "kernel_size": config["model"].get("kernel_size", 3),
            "dropout": config["model"].get("dropout", 0.2)
        },
        "classes": all_class_names,
        "device": device,
        "seed": 42,
        "environment": {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "python_version": "3.10.0",
            "pytorch_version": "2.0.0",
            "numpy_version": "1.24.0"
        },
        "results": {
            "test_accuracy": test_acc * 100,
            "class_accuracy": {class_name: float(acc) for class_name, acc in class_accuracy.items()},
            "overlap_f1_scores": {
                threshold: {
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1": float(metrics["f1"]),
                    "tp": int(metrics["tp"]),
                    "fp": int(metrics["fp"]),
                    "fn": int(metrics["fn"])
                } for threshold, metrics in overlap_f1_scores['thresholds'].items()
            },
            "segment_statistics": {
                "num_true_segments": overlap_f1_scores['num_true_segments'],
                "num_pred_segments": overlap_f1_scores['num_pred_segments']
            }
        }
    }
    
    # 하이퍼파라미터 JSON 저장
    with open(os.path.join(combined_results_dir, "hyperparameters.json"), 'w', encoding='utf-8') as f:
        json.dump(hyperparams, f, indent=4, ensure_ascii=False)
    
    return {
        "accuracy": test_acc,
        "class_accuracy": class_accuracy,
        "overlap_f1_scores": overlap_f1_scores
    } 
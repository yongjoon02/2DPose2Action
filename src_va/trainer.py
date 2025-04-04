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
from sklearn.metrics import average_precision_score

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

def calculate_segmental_edit_score(true_labels, pred_labels):
    """
    Segmental Edit Score 계산
    
    Args:
        true_labels: 실제 레이블 시퀀스 (N,)
        pred_labels: 예측 레이블 시퀀스 (N,)
    
    Returns:
        edit_score: 정규화된 edit score (0~100)
        segments_info: 세그먼트 정보 딕셔너리
    """
    def get_segments(labels):
        """레이블 시퀀스를 세그먼트로 변환"""
        segments = []
        if len(labels) == 0:
            return segments
            
        current_label = labels[0]
        start_idx = 0
        
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                segments.append((start_idx, i-1, current_label))
                current_label = labels[i]
                start_idx = i
        
        segments.append((start_idx, len(labels)-1, current_label))
        return segments
    
    def levenshtein_distance(s1, s2):
        """두 시퀀스 간의 레벤슈타인 거리 계산"""
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    # 세그먼트 추출
    true_segments = get_segments(true_labels)
    pred_segments = get_segments(pred_labels)
    
    # 세그먼트 레이블 시퀀스 생성
    true_segment_labels = [seg[2] for seg in true_segments]
    pred_segment_labels = [seg[2] for seg in pred_segments]
    
    # Edit distance 계산
    edit_distance = levenshtein_distance(true_segment_labels, pred_segment_labels)
    
    # Score 정규화 (0~100)
    max_distance = max(len(true_segment_labels), len(pred_segment_labels))
    if max_distance == 0:
        edit_score = 100.0
    else:
        edit_score = (1 - edit_distance / max_distance) * 100
    
    # 세그먼트 정보 수집
    segments_info = {
        "true_segments": true_segments,
        "pred_segments": pred_segments,
        "num_true_segments": len(true_segments),
        "num_pred_segments": len(pred_segments),
        "edit_distance": edit_distance,
        "normalized_score": edit_score
    }
    
    return edit_score, segments_info

def calculate_map(true_labels, pred_scores):
    """
    평균 정밀도(Average Precision)와 mAP(mean Average Precision) 계산
    
    Args:
        true_labels: 실제 레이블 (one-hot encoding 아님, 클래스 인덱스) - shape: (N,)
        pred_scores: 각 클래스에 대한 예측 확률 - shape: (N, num_classes)
    
    Returns:
        mAP: 모든 클래스에 대한 평균 AP
        class_ap: 각 클래스별 AP 딕셔너리
    """
    # 클래스 수 확인
    num_classes = pred_scores.shape[1]
    
    # 원핫 인코딩으로 변환
    y_true_onehot = np.zeros((len(true_labels), num_classes))
    for i, label in enumerate(true_labels):
        y_true_onehot[i, label] = 1
    
    # 각 클래스별 AP 계산
    class_ap = {}
    valid_ap_sum = 0.0
    valid_classes = 0
    
    for c in range(num_classes):
        # 클래스 c가 데이터셋에 존재하는지 확인
        if np.sum(y_true_onehot[:, c]) > 0:
            ap = average_precision_score(y_true_onehot[:, c], pred_scores[:, c])
            class_ap[c] = ap
            valid_ap_sum += ap
            valid_classes += 1
    
    # mAP 계산 - 유효한 클래스에 대해서만 평균 계산
    mAP = valid_ap_sum / valid_classes if valid_classes > 0 else 0.0
    
    return mAP, class_ap

def calculate_overlap_f1(true_labels, pred_labels, thresholds=[0.25, 0.5]):
    """
    시간 구간 기반의 F1 점수를 계산합니다.
    
    Args:
        true_labels: 실제 레이블 리스트
        pred_labels: 예측 레이블 리스트
        thresholds: 평가할 겹침 비율 임계값 리스트
    
    Returns:
        각 임계값에 대한 결과를 담은 딕셔너리
    """
    # 세그먼트 추출 (같은 레이블이 연속적으로 나타나는 구간)
    def extract_segments(labels):
        segments = []
        if len(labels) == 0:
            return segments
            
        current_label = labels[0]
        start_idx = 0
        
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                segments.append((start_idx, i-1, current_label))
                current_label = labels[i]
                start_idx = i
        
        segments.append((start_idx, len(labels)-1, current_label))
        return segments
    
    # 실제 세그먼트와 예측 세그먼트 추출
    true_segments = extract_segments(true_labels)
    pred_segments = extract_segments(pred_labels)
    
    num_true_segments = len(true_segments)
    num_pred_segments = len(pred_segments)
    
    # 클래스별로 세그먼트 분리
    true_by_class = {}
    pred_by_class = {}
    
    for t_start, t_end, t_class in true_segments:
        if t_class not in true_by_class:
            true_by_class[t_class] = []
        true_by_class[t_class].append((t_start, t_end, t_class))
    
    for p_start, p_end, p_class in pred_segments:
        if p_class not in pred_by_class:
            pred_by_class[p_class] = []
        pred_by_class[p_class].append((p_start, p_end, p_class))
    
    # 모든 클래스 목록
    all_classes = sorted(set(list(true_by_class.keys()) + list(pred_by_class.keys())))
    
    results = {}
    
    for threshold in thresholds:
        threshold_str = str(threshold)
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for cls in all_classes:
            true_segs = true_by_class.get(cls, [])
            pred_segs = pred_by_class.get(cls, [])
            
            # 매칭된 예측 세그먼트 추적
            matched_preds = set()
            
            for i, (t_start, t_end, _) in enumerate(true_segs):
                t_duration = t_end - t_start + 1
                best_overlap = 0
                best_pred_idx = None
                
                for j, (p_start, p_end, _) in enumerate(pred_segs):
                    if j in matched_preds:
                        continue
                    
                    # 겹치는 구간 계산
                    overlap_start = max(t_start, p_start)
                    overlap_end = min(t_end, p_end)
                    
                    if overlap_start <= overlap_end:
                        overlap_duration = overlap_end - overlap_start + 1
                        p_duration = p_end - p_start + 1
                        
                        # IoU(Intersection over Union) 계산
                        union_duration = t_duration + p_duration - overlap_duration
                        overlap_ratio = overlap_duration / union_duration
                        
                        if overlap_ratio > best_overlap:
                            best_overlap = overlap_ratio
                            best_pred_idx = j
                
                # 임계값을 초과하는 충분한 겹침이 있으면 TP로 간주
                if best_overlap >= threshold:
                    total_tp += 1
                    matched_preds.add(best_pred_idx)
                else:
                    total_fn += 1
            
            # 매칭되지 않은 예측은 FP로 간주
            total_fp += len(pred_segs) - len(matched_preds)
        
        # 지표 계산
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[threshold_str] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn
        }
    
    return {
        'thresholds': results,
        'num_true_segments': num_true_segments,
        'num_pred_segments': num_pred_segments
    }

def evaluate_and_save_test_results(model, test_loader, criterion, config, result_dir):
    """
    테스트 데이터셋에 대한 모델 평가 및 결과 저장
    """
    device = config["training"]["device"]
    model.eval()
    
    # 테스트 세트에 대한 평가
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
    
    # 확률 점수 계산을 위한 변수 준비
    all_true_labels = []
    all_pred_scores = []
    
    # 모델 예측 확률 얻기
    with torch.no_grad():
        for inputs, labels in test_loader:
            batch_size = len(inputs)
            
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
                
                # 소프트맥스 적용하여 확률 계산
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
                # 확률 점수와 레이블 저장
                all_true_labels.extend(seq_labels.cpu().numpy())
                all_pred_scores.extend(probs.cpu().numpy())
    
    # 리스트를 배열로 변환
    all_true_labels = np.array(all_true_labels)
    all_pred_scores = np.array(all_pred_scores)
    
    # mAP 계산
    mAP, class_ap = calculate_map(all_true_labels, all_pred_scores)
    
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
    
    # Segmental Edit Score 계산
    edit_score, segment_info = calculate_segmental_edit_score(test_labels, test_preds)
    
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
    
    # Segmental Edit Score 저장
    with open(os.path.join(combined_results_dir, "segmental_edit_score.txt"), 'w') as f:
        f.write("Segmental Edit Score 결과\n\n")
        f.write(f"총 세그먼트 수: 실제={segment_info['num_true_segments']}, 예측={segment_info['num_pred_segments']}\n")
        f.write(f"Edit Distance: {segment_info['edit_distance']}\n")
        f.write(f"Normalized Edit Score: {edit_score:.4f}\n")
    
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
    
    # mAP 결과 저장
    with open(os.path.join(combined_results_dir, "map_results.txt"), 'w') as f:
        f.write(f"mAP: {mAP:.4f}\n\n")
        f.write("클래스별 AP:\n")
        for class_idx, ap in class_ap.items():
            class_name = all_class_names[class_idx]
            f.write(f"{class_name}: {ap:.4f}\n")
    
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
            "mAP": float(mAP * 100),  # 계산된 mAP 사용
            "class_ap": {all_class_names[c]: float(ap * 100) for c, ap in class_ap.items()},
            "overlap_f1_scores": {
                str(int(float(threshold)*100)): {
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1": float(metrics["f1"]),
                    "tp": int(metrics["tp"]),
                    "fp": int(metrics["fp"]),
                    "fn": int(metrics["fn"])
                } for threshold, metrics in overlap_f1_scores['thresholds'].items()
            },
            "segmental_edit_score": float(edit_score),
            "segment_statistics": {
                "num_true_segments": int(segment_info["num_true_segments"]),
                "num_pred_segments": int(segment_info["num_pred_segments"]),
                "edit_distance": int(segment_info["edit_distance"])
            }
        }
    }
    
    # 하이퍼파라미터 JSON 저장
    with open(os.path.join(combined_results_dir, "hyperparameters.json"), 'w', encoding='utf-8') as f:
        json.dump(hyperparams, f, indent=4, ensure_ascii=False)
    
    return {
        "accuracy": test_acc,
        "class_accuracy": class_accuracy,
        "mAP": mAP,
        "class_ap": class_ap,
        "overlap_f1_scores": overlap_f1_scores,
        "segmental_edit_score": edit_score,
        "segment_statistics": segment_info
    } 
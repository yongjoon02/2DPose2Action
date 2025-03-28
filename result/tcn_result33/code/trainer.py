import os
import torch
import numpy as np
import json
import csv
import time
import shutil
import glob
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib

from .models import TCN
from .losses import WeightedFocalLoss
from .datasets import class_mapping

matplotlib.use('Agg')  # 반드시 다른 matplotlib 코드보다 먼저 실행되게 하세요

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(dataloader, desc="훈련 중"):
        coords, labels, _ = batch
        coords = coords.to(device, dtype=torch.float32)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(coords, conservative_no_activity=False, apply_transition_rules=False)
        
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
            
            outputs = model(coords, conservative_no_activity=False, apply_transition_rules=False)
            
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
    
    # 전체 정확도 계산
    accuracy = 100 * correct / total if total > 0 else 0
    
    return epoch_loss / len(dataloader), accuracy

def train_with_early_stopping(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, 
                           num_epochs=200, patience=20, fold_dir=None):
    best_valid_acc = 0
    best_model_path = os.path.join(fold_dir, "best_model.pth")
    counter = 0
    
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
    
    if fold_dir:
        save_learning_curves(
            train_loss_history,
            valid_loss_history,
            valid_acc_history,
            os.path.join(fold_dir, "learning_curves.csv")
        )
    
    model.load_state_dict(torch.load(best_model_path))
    return model, best_valid_acc

def create_result_dir():
    """결과 저장을 위한 고유한 디렉토리 이름 생성"""
    base_dir = "result"
    base_name = "tcn_result"
    
    # 기본 디렉토리가 없으면 생성
    os.makedirs(base_dir, exist_ok=True)
    
    # 기존 tcn_result 디렉토리 확인
    existing_dirs = glob.glob(os.path.join(base_dir, f"{base_name}*"))
    
    if not existing_dirs:
        result_dir = os.path.join(base_dir, base_name)
    else:
        # 숫자 접미사가 있는 디렉토리들 찾기
        indices = []
        for dir_path in existing_dirs:
            dir_name = os.path.basename(dir_path)
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

def save_source_code(dest_dir):
    """현재 사용 중인 코드 파일들을 저장"""
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 코드 디렉토리 생성 (src_files 대신 code 폴더 사용)
    code_dir = os.path.join(dest_dir, "code")
    os.makedirs(code_dir, exist_ok=True)
    
    # src 디렉토리 내의 Python 파일들
    for file in glob.glob(os.path.join(src_dir, "src", "*.py")):
        shutil.copy2(file, code_dir)
    
    # train.py 파일 복사
    train_py = os.path.join(src_dir, "script", "train.py")
    if os.path.exists(train_py):
        shutil.copy2(train_py, code_dir)
    
    print(f"소스 코드가 {code_dir}에 저장되었습니다.")

def generate_evaluation_results(model, dataloader, device, fold_dir, class_names):
    """모델 평가 및 결과 파일 생성"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="평가 중"):
            coords, labels, _ = batch
            coords = coords.to(device, dtype=torch.float32)
            
            outputs = model(coords, conservative_no_activity=True, apply_transition_rules=True)
            if isinstance(outputs, tuple):
                predictions = outputs[1]  # 후처리된 예측값
            else:
                # 후처리가 적용되지 않은 경우
                outputs = outputs
                _, predictions = torch.max(outputs, dim=-1)
            
            # 배치 차원과 시퀀스 차원을 평평하게
            batch_preds = predictions.detach().cpu().numpy().flatten()
            batch_labels = labels.detach().cpu().numpy().flatten()
            
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
    
    # NumPy 배열로 변환
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 혼동 행렬 계산
    cm = confusion_matrix(all_labels, all_preds)
    
    # 혼동 행렬 저장 (CSV)
    cm_path = os.path.join(fold_dir, "confusion_matrix.csv")
    with open(cm_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + class_names)
        for i, row in enumerate(cm):
            writer.writerow([class_names[i]] + list(row))
    
    # 혼동 행렬 시각화
    save_confusion_matrix_to_png(
        cm, 
        class_names,
        os.path.join(fold_dir, "confusion_matrix.png")
    )
    
    # 분류 보고서
    report = classification_report(all_labels, all_preds, 
                                  target_names=class_names, 
                                  digits=4)
    
    with open(os.path.join(fold_dir, "classification_report.txt"), 'w') as f:
        f.write(report)
    
    # 클래스별 정확도
    class_acc = {}
    for i, class_name in enumerate(class_names):
        mask = (all_labels == i)
        if np.sum(mask) > 0:  # 해당 클래스의 샘플이 있는 경우만
            acc = np.mean(all_preds[mask] == i) * 100
            class_acc[class_name] = acc
    
    # 클래스별 정확도 저장
    with open(os.path.join(fold_dir, "class_accuracy.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'accuracy'])
        for class_name, acc in class_acc.items():
            writer.writerow([class_name, f"{acc:.2f}%"])
    
    return cm, report, class_acc

def combine_evaluation_results(fold_dirs, combined_results_dir, class_names):
    """모든 폴드의 평가 결과를 결합"""
    # 모든 폴드의 혼동 행렬 합치기
    combined_cm = None
    combined_class_acc = {class_name: [] for class_name in class_names}
    
    for fold_dir in fold_dirs:
        # 혼동 행렬 로드
        cm_path = os.path.join(fold_dir, "confusion_matrix.csv")
        if os.path.exists(cm_path):
            cm = []
            with open(cm_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # 헤더 건너뛰기
                for row in reader:
                    cm.append([int(x) for x in row[1:]])
            cm = np.array(cm)
            
            if combined_cm is None:
                combined_cm = cm
            else:
                combined_cm += cm
        
        # 클래스별 정확도 로드
        acc_path = os.path.join(fold_dir, "class_accuracy.csv")
        if os.path.exists(acc_path):
            with open(acc_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # 헤더 건너뛰기
                for row in reader:
                    class_name, acc = row
                    acc = float(acc.strip('%'))
                    combined_class_acc[class_name].append(acc)
    
    # 결합된 혼동 행렬 저장
    if combined_cm is not None:
        # CSV 저장
        cm_path = os.path.join(combined_results_dir, "combined_confusion_matrix.csv")
        with open(cm_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([''] + class_names)
            for i, row in enumerate(combined_cm):
                writer.writerow([class_names[i]] + list(row))
        
        # 시각화
        save_confusion_matrix_to_png(
            combined_cm, 
            class_names,
            os.path.join(combined_results_dir, "confusion_matrix.png")
        )
    
    # 결합된 클래스별 정확도 계산 및 저장
    class_acc_path = os.path.join(combined_results_dir, "class_accuracy.csv")
    with open(class_acc_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'mean_accuracy', 'std_accuracy'])
        for class_name, accs in combined_class_acc.items():
            if accs:  # 리스트가 비어있지 않은 경우만
                mean_acc = np.mean(accs)
                std_acc = np.std(accs)
                writer.writerow([class_name, f"{mean_acc:.2f}%", f"{std_acc:.2f}%"])
    
    # 결합된 분류 보고서 생성
    if combined_cm is not None:
        # 클래스별 precision, recall, f1-score 계산
        precisions = np.zeros(len(class_names))
        recalls = np.zeros(len(class_names))
        f1_scores = np.zeros(len(class_names))
        
        for i in range(len(class_names)):
            # precision = TP / (TP + FP)
            tp = combined_cm[i, i]
            col_sum = np.sum(combined_cm[:, i])
            precisions[i] = tp / col_sum if col_sum > 0 else 0
            
            # recall = TP / (TP + FN)
            row_sum = np.sum(combined_cm[i, :])
            recalls[i] = tp / row_sum if row_sum > 0 else 0
            
            # f1-score = 2 * (precision * recall) / (precision + recall)
            if precisions[i] + recalls[i] > 0:
                f1_scores[i] = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])
            else:
                f1_scores[i] = 0
        
        # 전체 정확도
        total_samples = np.sum(combined_cm)
        accuracy = np.sum(np.diag(combined_cm)) / total_samples if total_samples > 0 else 0
        
        # 분류 보고서 저장
        report_path = os.path.join(combined_results_dir, "classification_report.txt")
        with open(report_path, 'w') as f:
            f.write("Combined Classification Report\n\n")
            f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-score':<12} {'Support':<12}\n")
            f.write("-" * 60 + "\n")
            
            for i, class_name in enumerate(class_names):
                support = np.sum(combined_cm[i, :])
                f.write(f"{class_name:<15} {precisions[i]:.4f}       {recalls[i]:.4f}       ")
                f.write(f"{f1_scores[i]:.4f}       {support:<12}\n")
            
            f.write("\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Macro Avg: {np.mean(precisions):.4f}    {np.mean(recalls):.4f}    {np.mean(f1_scores):.4f}    {total_samples}\n")
            
            # 가중 평균 계산
            weights = np.sum(combined_cm, axis=1) / total_samples
            w_precision = np.sum(precisions * weights)
            w_recall = np.sum(recalls * weights)
            w_f1 = np.sum(f1_scores * weights)
            f.write(f"Weighted Avg: {w_precision:.4f}    {w_recall:.4f}    {w_f1:.4f}    {total_samples}\n")

def process_predictions(pred_sequence):
    """예측 시퀀스를 JSON 형식으로 변환 - 연속된 같은 활동을 프레임 범위로 그룹화"""
    # 클래스 인덱스와 이름 매핑
    class_indices = {i: name for i, name in enumerate(class_mapping.values())}
    
    if len(pred_sequence) == 0:
        return []
    
    # 연속된 같은 활동을 범위로 그룹화
    current_class = pred_sequence[0]
    start_frame = 0
    result = []
    
    for frame, pred in enumerate(pred_sequence):
        if pred != current_class:
            # 이전 범위 저장
            result.append({
                "frameRange": [int(start_frame), int(frame - 1)],
                "activity": class_indices[int(current_class)]
            })
            # 새 범위 시작
            current_class = pred
            start_frame = frame
    
    # 마지막 범위 추가
    result.append({
        "frameRange": [int(start_frame), int(len(pred_sequence) - 1)],
        "activity": class_indices[int(current_class)]
    })
    
    return result

def calculate_overlap_f1(true_activities, pred_activities, overlap_thresholds=[0.25, 0.5]):
    """
    시간 구간 기반의 F1 점수를 계산합니다.
    
    Args:
        true_activities: 실제 활동 리스트 - 각 항목은 [시작프레임, 종료프레임, 활동클래스] 형식
        pred_activities: 예측 활동 리스트 - 각 항목은 [시작프레임, 종료프레임, 활동클래스] 형식
        overlap_thresholds: 평가할 겹침 비율 임계값 리스트
    
    Returns:
        각 임계값에 대한 precision, recall, f1 결과를 담은 딕셔너리
    """
    results = {}
    
    for threshold in overlap_thresholds:
        tp = 0
        fp = 0
        fn = 0
        
        # 각 실제 활동에 대해 매칭된 예측을 추적
        matched_preds = set()
        
        # 각 실제 활동에 대해
        for t_start, t_end, t_class in true_activities:
            t_duration = t_end - t_start + 1
            best_overlap = 0
            best_pred_idx = None
            
            # 모든 예측 활동과 비교
            for i, (p_start, p_end, p_class) in enumerate(pred_activities):
                # 이미 매칭된 예측은 건너뜀
                if i in matched_preds:
                    continue
                
                # 같은 클래스인 경우만 계산
                if t_class != p_class:
                    continue
                
                # 겹치는 구간 계산
                overlap_start = max(t_start, p_start)
                overlap_end = min(t_end, p_end)
                
                if overlap_start <= overlap_end:
                    overlap_duration = overlap_end - overlap_start + 1
                    overlap_ratio = min(
                        overlap_duration / t_duration,
                        overlap_duration / (p_end - p_start + 1)
                    )
                    
                    if overlap_ratio > best_overlap:
                        best_overlap = overlap_ratio
                        best_pred_idx = i
            
            # 임계값을 초과하는 충분한 겹침이 있으면 TP로 간주
            if best_overlap >= threshold:
                tp += 1
                matched_preds.add(best_pred_idx)
            else:
                fn += 1  # 매칭되는 예측 활동이 없음
        
        # 매칭되지 않은 예측 활동은 FP로 간주
        fp = len(pred_activities) - len(matched_preds)
        
        # 지표 계산
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[threshold] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    return results

def extract_activity_segments(labels, frames=None):
    """
    연속 프레임 레이블을 활동 구간 리스트로 변환합니다.
    
    Args:
        labels: 각 프레임의 활동 레이블 배열
        frames: 프레임 번호 배열 (없으면 0부터 시작)
    
    Returns:
        활동 구간 리스트 - 각 항목은 [시작프레임, 종료프레임, 활동클래스] 형식
    """
    if frames is None:
        frames = np.arange(len(labels))
    
    segments = []
    current_class = labels[0]
    start_frame = frames[0]
    
    for i in range(1, len(labels)):
        if labels[i] != current_class:
            segments.append([start_frame, frames[i-1], current_class])
            current_class = labels[i]
            start_frame = frames[i]
    
    # 마지막 세그먼트 추가
    segments.append([start_frame, frames[-1], current_class])
    
    return segments

def test_ensemble_with_overlap_f1(models, dataloader, device, output_dir, combined_results_dir, config, final_results):
    """
    앙상블 모델에 대해 테스트를 수행하고 Overlap F1 Score를 계산합니다.
    """
    for model in models:
        model.eval()
    
    print(f"테스트 결과를 {output_dir}에 저장합니다.")
    os.makedirs(output_dir, exist_ok=True)
    
    all_labels = []
    all_preds = []
    all_true_segments = []
    all_pred_segments = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="테스트 중"):
            coords, labels, filenames = batch
            coords = coords.to(device, dtype=torch.float32)
            
            # 모든 모델의 예측 결과 수집
            ensemble_preds = []
            
            for model in models:
                outputs = model(coords, conservative_no_activity=True, apply_transition_rules=True)
                
                if isinstance(outputs, tuple):
                    _, predictions = outputs
                else:
                    _, predictions = torch.max(outputs, dim=-1)
                
                ensemble_preds.append(predictions)
            
            # 앙상블 결과 계산 (다수결 방식)
            ensemble_preds = torch.stack(ensemble_preds)
            ensemble_preds = torch.mode(ensemble_preds, dim=0).values
            
            # 각 배치 항목에 대해 처리
            for i in range(coords.shape[0]):
                pred_labels = ensemble_preds[i].cpu().numpy()
                
                # 레이블이 있는 경우 실제 레이블도 저장
                if labels is not None:
                    true_labels = labels[i].cpu().numpy()
                    
                    # 활동 구간 추출
                    pred_segments = extract_activity_segments(pred_labels)
                    true_segments = extract_activity_segments(true_labels)
                    
                    all_true_segments.extend(true_segments)
                    all_pred_segments.extend(pred_segments)
                
                # 프레임 단위 정확도 계산을 위해 저장
                if labels is not None:
                    all_labels.append(labels[i].reshape(-1))
                    all_preds.append(ensemble_preds[i].reshape(-1))
                
                # 파일별 결과 처리 (기존 코드와 동일)
                json_result = process_predictions(pred_labels)
                base_filename = os.path.basename(filenames[i])
                json_filename = base_filename.replace('.csv', '_prediction.json')
                json_path = os.path.join(output_dir, json_filename)
                
                try:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_result, f, indent=4, ensure_ascii=False)
                except Exception as e:
                    print(f"파일 저장 오류: {json_path}, 에러: {e}")
    
    # 테스트 성능 측정 (라벨이 있는 경우만)
    if all_labels:
        # 프레임 단위 평가 (기존 코드)
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
        all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
        
        # 기존 정확도 계산 및 시각화 코드...
        # (기존 코드 생략)
        
        # Overlap F1 Score 계산
        overlap_results = calculate_and_save_overlap_f1(
            all_true_segments, 
            all_pred_segments,
            combined_results_dir
        )
        
        # Overlap F1 Score 결과 저장
        with open(os.path.join(combined_results_dir, "overlap_f1_results.txt"), "w", encoding="utf-8") as f:
            f.write("Overlap F1 Score 결과\n\n")
            for threshold, metrics in overlap_results.items():
                f.write(f"임계값 {threshold*100:.0f}%:\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
                f.write(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}\n\n")
        
        # CSV 형식으로도 저장
        with open(os.path.join(combined_results_dir, "overlap_f1_results.csv"), "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(['Threshold', 'Precision', 'Recall', 'F1', 'TP', 'FP', 'FN'])
            for threshold, metrics in overlap_results.items():
                writer.writerow([
                    f"{threshold*100:.0f}%", 
                    f"{metrics['precision']:.4f}", 
                    f"{metrics['recall']:.4f}", 
                    f"{metrics['f1']:.4f}",
                    metrics['tp'],
                    metrics['fp'],
                    metrics['fn']
                ])
        
        # 결과를 final_results에 추가
        final_results["overlap_f1_scores"] = {
            str(int(threshold*100)): {"f1": metrics["f1"]} 
            for threshold, metrics in overlap_results.items()
        }
        
        print(f"Overlap F1 Score 결과:")
        for threshold, metrics in overlap_results.items():
            print(f"  임계값 {threshold*100:.0f}%: F1={metrics['f1']:.4f} (P={metrics['precision']:.4f}, R={metrics['recall']:.4f})")

def test(model, dataloader, device, save_dir=None, dataset_path=None, return_predictions=False):
    """테스트 데이터에 대한 모델 평가 및 예측 결과 생성"""
    model.eval()
    all_preds = []
    all_labels = []
    all_json_results = {}  # JSON 결과 저장용
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="테스트 중"):
            coords, labels, filenames = batch
            coords = coords.to(device, dtype=torch.float32)
            
            # 모델 예측
            outputs = model(coords, conservative_no_activity=True, apply_transition_rules=True)
            
            # 튜플로 반환되는 경우 처리
            if isinstance(outputs, tuple):
                logits, predictions = outputs
            else:
                logits = outputs
                _, predictions = torch.max(logits, dim=-1)
            
            # 파일별 결과 처리
            for i, filename in enumerate(filenames):
                # 예측 결과를 JSON 형식으로 변환
                pred_sequence = predictions[i].cpu().numpy()
                json_result = process_predictions(pred_sequence)
                
                # 결과 저장
                all_json_results[filename] = json_result
                
                # 라벨과 예측값 수집
                if labels is not None:
                    all_preds.extend(predictions[i].cpu().numpy())
                    all_labels.extend(labels[i].cpu().numpy())
    
    # 결과 저장
    if save_dir and dataset_path:
        # output_json 폴더를 다시 생성하지 않고 직접 save_dir에 저장
        json_dir = save_dir
        
        for filename, result in all_json_results.items():
            # 원본 파일 이름에서 .csv 제거하고 _prediction.json 추가
            json_filename = os.path.basename(filename).replace('.csv', '_prediction.json')
            json_path = os.path.join(json_dir, json_filename)
            
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=4)
    
    # 평가 지표 계산
    metrics = {}
    if all_labels and all_preds:
        # 정확도 계산
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
        metrics["accuracy"] = accuracy
        
        # 혼동 행렬
        cm = confusion_matrix(all_labels, all_preds)
        metrics["confusion_matrix"] = cm.tolist()
    
    if return_predictions:
        return metrics, all_json_results
    else:
        return metrics

def train_model(dataset, config):
    device = torch.device(config["device"])
    
    # 결과 저장 디렉토리 생성
    result_dir = create_result_dir()
    combined_results_dir = os.path.join(result_dir, "combined_results")
    os.makedirs(combined_results_dir, exist_ok=True)
    
    # output_json 폴더 생성
    output_json_dir = os.path.join(combined_results_dir, "output_json")
    os.makedirs(output_json_dir, exist_ok=True)
    
    # 소스 코드 저장
    save_source_code(result_dir)
    
    # 하이퍼파라미터에 세부 정보 추가
    if "augmentation_details" not in config:
        config["augmentation_details"] = {
            "rotation_range": 10,  # 회전 각도 범위
            "shift_range": 0.1,    # 이동 범위
            "scale_range": 0.1,    # 스케일 범위
            "random_flip": True,   # 무작위 뒤집기 적용 여부
            "random_jitter": True, # 무작위 지터링 적용 여부
            "jitter_scale": 0.05,  # 지터링 스케일
            "time_masking": True,  # 시간 마스킹 적용 여부
            "time_mask_size": 5    # 시간 마스크 크기
        }
    
    if "inference_details" not in config:
        config["inference_details"] = {
            "confidence_threshold": 0.5,  # 신뢰도 임계값
            "apply_smoothing": True,      # 예측 결과 스무딩 적용 여부
            "smoothing_window": 5,        # 스무딩 윈도우 크기
            "ensemble_method": "majority_voting"  # 앙상블 방법
        }
    
    # 하이퍼파라미터 저장
    with open(os.path.join(combined_results_dir, "hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    # 데이터셋의 레이블 추출
    all_labels = [label[0] for _, label, _ in dataset]  # 첫 프레임의 레이블만 사용
    
    # 7:1:2 비율로 데이터 분할 (train+val:test = 8:2 먼저 분할 후, train:val = 7:1 분할)
    train_val_idx, test_idx = train_test_split(
        np.arange(len(dataset)), 
        test_size=0.2,  # 20%는 테스트용
        random_state=config["seed"],
        stratify=all_labels  # 클래스 균형 유지
    )
    
    # train_val 데이터 중에서 train:val = 7:1 (즉, 87.5%:12.5%)로 분할
    # all_labels에서 해당 인덱스만 선택하여 stratify에 사용
    selected_labels = [all_labels[i] for i in train_val_idx]
    
    # K-fold 교차 검증 준비 (train_val 데이터에 대해서만)
    kfold = StratifiedKFold(n_splits=config["training"]["num_folds"], shuffle=True, random_state=config["seed"])
    
    fold_accuracies = []
    fold_losses = []
    fold_results = []
    fold_dirs = []
    
    all_models = []  # 모든 폴드의 모델을 저장할 리스트
    
    # 테스트 데이터로더 미리 생성
    test_sampler = SubsetRandomSampler(test_idx)
    test_loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], sampler=test_sampler)
    
    print(f"데이터셋 분할: 전체 {len(dataset)}개 중 학습+검증용 {len(train_val_idx)}개, 테스트용 {len(test_idx)}개")
    
    for fold, (train_fold_idx, val_fold_idx) in enumerate(kfold.split(np.zeros(len(train_val_idx)), selected_labels), 1):
        print(f"\nFold {fold} 시작")
        fold_dir = os.path.join(result_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        fold_dirs.append(fold_dir)
        
        # train_val_idx에서 실제 인덱스 가져오기
        train_idx = [train_val_idx[i] for i in train_fold_idx]
        val_idx = [train_val_idx[i] for i in val_fold_idx]
        
        print(f"Fold {fold} 데이터 분할: 학습용 {len(train_idx)}개, 검증용 {len(val_idx)}개")
        
        # 데이터로더 생성
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], 
                                sampler=train_sampler)
        valid_loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], 
                                sampler=valid_sampler)
        
        # 모델 초기화
        model = TCN(
            input_size=config["model"]["input_size"],
            output_size=config["model"]["output_size"],
            num_channels=config["model"]["hidden_channels"],
            kernel_size=config["model"]["kernel_size"],
            dropout=config["model"]["dropout"],
            use_se=config["model"]["use_se"]
        ).to(device)
        
        # 손실 함수와 옵티마이저 설정
        loss_params = {"gamma": config["training"]["loss_params"]["gamma"]}
        
        # 클래스 가중치가 있으면 weight_dict로 전달
        if "class_weights" in config["training"]["loss_params"] and config["training"]["loss_params"]["class_weights"] is not None:
            weight_dict = config["training"]["loss_params"]["class_weights"]
            # 클래스 매핑을 사용하여 인덱스 기반 가중치 배열 생성
            class_names = list(class_mapping.values())
            weight_tensor = torch.ones(len(class_names), device=device)
            for i, class_name in enumerate(class_names):
                if class_name in weight_dict:
                    weight_tensor[i] = weight_dict[class_name]
            
            # 가중치 텐서를 weight 매개변수로 전달
            loss_params["weight"] = weight_tensor
        
        criterion = WeightedFocalLoss(**loss_params)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )
        
        # 스케줄러 설정
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config["training"]["scheduler_params"]["T_0"],
            T_mult=config["training"]["scheduler_params"]["T_mult"]
        )
        
        # 모델 학습
        model, best_acc = train_with_early_stopping(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=config["training"]["num_epochs"],
            patience=config["training"]["early_stopping_patience"],
            fold_dir=fold_dir
        )
        
        # 최종 검증 손실 계산
        final_valid_loss, final_valid_acc = validate(model, valid_loader, criterion, device)
        
        fold_accuracies.append(best_acc)
        fold_losses.append(final_valid_loss)
        fold_results.append({
            "fold": fold,
            "loss": final_valid_loss,
            "accuracy": best_acc
        })
        
        print(f"Fold {fold} 완료 - 최고 검증 정확도: {best_acc:.2f}%, 손실: {final_valid_loss:.6f}")
        
        # 모델 저장 (각 폴드별로 저장)
        torch.save(model.state_dict(), os.path.join(fold_dir, "final_model.pth"))
        
        # 통합 결과를 위해 모델도 combined_results에 복사
        combined_model_path = os.path.join(combined_results_dir, f"model_fold_{fold}.pth")
        shutil.copy(os.path.join(fold_dir, "final_model.pth"), combined_model_path)
        
        all_models.append(model)
    
    # 모든 폴드의 평균 결과 계산
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_loss = np.mean(fold_losses)
    std_loss = np.std(fold_losses)
    
    print(f"\n교차 검증 결과:")
    print(f"평균 정확도: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"평균 손실: {mean_loss:.6f} ± {std_loss:.6f}")
    
    # 결과 저장
    final_results = {
        "mean_val_loss": float(mean_loss),
        "std_val_loss": float(std_loss),
        "mean_val_accuracy": float(mean_acc),
        "std_val_accuracy": float(std_acc),
        "fold_results": fold_results
    }
    
    # 업데이트된 config에 results 추가
    config["results"] = final_results
    
    # 최종 하이퍼파라미터와 결과 저장
    with open(os.path.join(combined_results_dir, "hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    # 결과 요약 파일 저장 - UTF-8 인코딩 추가하여 한글 깨짐 방지
    with open(os.path.join(combined_results_dir, "results_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"평균 검증 정확도: {mean_acc:.2f}% ± {std_acc:.2f}%\n")
        f.write(f"평균 검증 손실: {mean_loss:.6f} ± {std_loss:.6f}\n\n")
        f.write("각 폴드 결과:\n")
        for i, result in enumerate(fold_results, 1):
            f.write(f"Fold {i}: 정확도 = {result['accuracy']:.2f}%, 손실 = {result['loss']:.6f}\n")
    
    # 최종 테스트 단계 - 모든 폴드 모델의 앙상블 사용
    print("\n테스트 데이터셋에 대한 평가 시작")
    
    # 테스트 데이터셋 처리를 위한 함수 구현
    def process_predictions(pred_sequence):
        """예측 시퀀스를 JSON 형식으로 변환 - 연속된 같은 활동을 프레임 범위로 그룹화"""
        # 클래스 인덱스와 이름 매핑
        class_indices = {i: name for i, name in enumerate(class_mapping.values())}
        
        if len(pred_sequence) == 0:
            return []
        
        # 연속된 같은 활동을 범위로 그룹화
        current_class = pred_sequence[0]
        start_frame = 0
        result = []
        
        for frame, pred in enumerate(pred_sequence):
            if pred != current_class:
                # 이전 범위 저장
                result.append({
                    "frameRange": [int(start_frame), int(frame - 1)],
                    "activity": class_indices[int(current_class)]
                })
                # 새 범위 시작
                current_class = pred
                start_frame = frame
        
        # 마지막 범위 추가
        result.append({
            "frameRange": [int(start_frame), int(len(pred_sequence) - 1)],
            "activity": class_indices[int(current_class)]
        })
        
        return result
    
    # 통합 결과만 저장하도록 테스트 함수 직접 구현
    def test_ensemble(models, dataloader, device, output_dir, combined_results_dir, config, final_results):
        for model in models:
            model.eval()
        
        print(f"테스트 결과를 {output_dir}에 저장합니다.")
        os.makedirs(output_dir, exist_ok=True)  # 디렉토리가 확실히 존재하는지 확인
        
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="테스트 중"):
                coords, labels, filenames = batch
                coords = coords.to(device, dtype=torch.float32)
                
                if labels is not None:
                    labels = labels.to(device)
                
                # 모든 모델의 예측 결과 수집
                ensemble_preds = []
                
                for model in models:
                    outputs = model(coords, conservative_no_activity=True, apply_transition_rules=True)
                    
                    if isinstance(outputs, tuple):
                        _, predictions = outputs
                    else:
                        _, predictions = torch.max(outputs, dim=-1)
                    
                    ensemble_preds.append(predictions)
                
                # 앙상블 결과 계산 (다수결 방식)
                ensemble_preds = torch.stack(ensemble_preds)
                ensemble_preds = torch.mode(ensemble_preds, dim=0).values
                
                # 레이블이 있는 경우 평가를 위해 저장
                if labels is not None:
                    # 배치와 시퀀스 차원을 평평하게
                    B, T = labels.shape
                    flat_labels = labels.reshape(-1)
                    flat_preds = ensemble_preds.reshape(-1)
                    
                    all_labels.append(flat_labels)
                    all_preds.append(flat_preds)
                
                # 파일별 결과 처리
                for i, filename in enumerate(filenames):
                    pred_sequence = ensemble_preds[i].cpu().numpy()
                    json_result = process_predictions(pred_sequence)
                    
                    # JSON 파일로 저장
                    base_filename = os.path.basename(filename)
                    json_filename = base_filename.replace('.csv', '_prediction.json')
                    json_path = os.path.join(output_dir, json_filename)
                    
                    try:
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(json_result, f, indent=4, ensure_ascii=False)
                    except Exception as e:
                        print(f"파일 저장 오류: {json_path}, 에러: {e}")
        
        # 테스트 성능 측정 (라벨이 있는 경우만)
        if all_labels:
            all_labels = torch.cat(all_labels, dim=0)
            all_preds = torch.cat(all_preds, dim=0)
            
            # 정확도 계산
            correct = (all_preds == all_labels).float()
            test_acc = correct.mean().item() * 100
            
            print(f"테스트 정확도: {test_acc:.2f}%")
            
            # 테스트 결과를 results_summary.txt에 추가
            with open(os.path.join(combined_results_dir, "results_summary.txt"), "a", encoding="utf-8") as f:
                f.write(f"\n테스트 정확도: {test_acc:.2f}%\n")
            
            # 테스트 결과를 final_results에 추가
            final_results["test_accuracy"] = float(test_acc)
            
            # 혼동 행렬 계산 및 저장
            cm = confusion_matrix(all_labels.cpu().numpy(), all_preds.cpu().numpy())
            class_names = config["classes"]
            
            # 혼동 행렬 CSV 저장
            cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
            cm_df.to_csv(os.path.join(combined_results_dir, "combined_confusion_matrix.csv"))
            
            # 분류 보고서 생성 및 저장
            report = classification_report(all_labels.cpu().numpy(), all_preds.cpu().numpy(), 
                                          target_names=class_names)
            with open(os.path.join(combined_results_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
                f.write(report)
            
            # 클래스별 정확도 계산 및 저장
            class_acc = {}
            for cls_idx, cls_name in enumerate(class_names):
                cls_mask = (all_labels == cls_idx)
                if cls_mask.sum() > 0:
                    cls_correct = correct[cls_mask]
                    class_acc[cls_name] = cls_correct.mean().item() * 100
                else:
                    class_acc[cls_name] = 0
            
            # 클래스별 정확도 CSV 저장
            class_acc_df = pd.DataFrame.from_dict(class_acc, orient='index', columns=['Accuracy'])
            class_acc_df.index.name = 'Class'
            class_acc_df.to_csv(os.path.join(combined_results_dir, "class_accuracy.csv"))
            
            # 업데이트된 config에 results 추가
            config["results"] = final_results
            
            # 최종 하이퍼파라미터와 결과 저장
            with open(os.path.join(combined_results_dir, "hyperparameters.json"), "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        
        print(f"테스트 결과가 {output_dir}에 저장되었습니다.")
    
    # 앙상블 테스트 실행
    test_ensemble(all_models, test_loader, device, output_json_dir, combined_results_dir, config, final_results)
    
    print(f"모든 결과가 {result_dir}에 저장되었습니다.")
    return result_dir, final_results

def save_learning_curves(train_losses, valid_losses, valid_accs, save_path):
    """학습 및 검증 손실과 정확도 내역을 CSV 파일로 저장합니다."""
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'valid_loss', 'valid_accuracy'])
        for epoch, (train_loss, valid_loss, valid_acc) in enumerate(zip(train_losses, valid_losses, valid_accs)):
            writer.writerow([epoch + 1, train_loss, valid_loss, valid_acc])

# 혼동 행렬 PNG 저장 확실히 처리
def save_confusion_matrix_to_png(cm, class_names, save_path):
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        
        # 저장 경로 디렉토리 확인
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        
        # 파일 저장 및 디버깅 메시지
        print(f"혼동 행렬 이미지 저장 시도: {save_path}")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"혼동 행렬 저장 성공: {save_path}")
        
        # 저장된 파일 확인
        if os.path.exists(save_path):
            print(f"파일 크기: {os.path.getsize(save_path)} 바이트")
        else:
            print(f"경고: 파일이 저장되지 않았습니다")
            
    except Exception as e:
        print(f"혼동 행렬 저장 오류: {str(e)}")
    finally:
        plt.close()

# F1 Overlap Score 계산 및 저장 함수 (디버깅 로그 추가)
def calculate_and_save_overlap_f1(true_segments, pred_segments, save_dir, thresholds=[0.25, 0.5]):
    print(f"F1 Overlap Score 계산 시작 (세그먼트 수: 실제={len(true_segments)}, 예측={len(pred_segments)})")
    
    results = {}
    for threshold in thresholds:
        print(f"임계값 {threshold*100:.0f}%에 대한 계산 시작")
        tp = 0
        fp = 0
        fn = 0
        
        matched_preds = set()
        
        # True Positives & False Negatives 계산
        for t_idx, (t_start, t_end, t_class) in enumerate(true_segments):
            t_duration = t_end - t_start + 1
            best_overlap = 0
            best_pred_idx = None
            
            for p_idx, (p_start, p_end, p_class) in enumerate(pred_segments):
                if p_idx in matched_preds or t_class != p_class:
                    continue
                
                overlap_start = max(t_start, p_start)
                overlap_end = min(t_end, p_end)
                
                if overlap_start <= overlap_end:
                    overlap_duration = overlap_end - overlap_start + 1
                    p_duration = p_end - p_start + 1
                    
                    overlap_ratio = min(
                        overlap_duration / t_duration,
                        overlap_duration / p_duration
                    )
                    
                    if overlap_ratio > best_overlap:
                        best_overlap = overlap_ratio
                        best_pred_idx = p_idx
            
            if best_overlap >= threshold:
                tp += 1
                matched_preds.add(best_pred_idx)
                print(f"  세그먼트 #{t_idx}: TP (겹침률={best_overlap:.2f})")
            else:
                fn += 1
                print(f"  세그먼트 #{t_idx}: FN (최대 겹침률={best_overlap:.2f})")
        
        # False Positives 계산
        fp = len(pred_segments) - len(matched_preds)
        
        # 지표 계산
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"임계값 {threshold*100:.0f}% 결과: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
        print(f"  TP={tp}, FP={fp}, FN={fn}")
        
        results[threshold] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    # 결과 저장 (텍스트 파일)
    try:
        txt_path = os.path.join(save_dir, "overlap_f1_results.txt")
        print(f"결과 텍스트 파일 저장 시도: {txt_path}")
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("F1 Overlap Score 결과\n\n")
            for threshold, metrics in results.items():
                f.write(f"임계값 {threshold*100:.0f}%:\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
                f.write(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}\n\n")
                
        print(f"텍스트 파일 저장 완료: {txt_path}")
        
        # CSV 파일로도 저장
        csv_path = os.path.join(save_dir, "overlap_f1_results.csv")
        print(f"CSV 파일 저장 시도: {csv_path}")
        
        with open(csv_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(['Threshold', 'Precision', 'Recall', 'F1', 'TP', 'FP', 'FN'])
            for threshold, metrics in results.items():
                writer.writerow([
                    f"{threshold*100:.0f}%", 
                    f"{metrics['precision']:.4f}", 
                    f"{metrics['recall']:.4f}", 
                    f"{metrics['f1']:.4f}",
                    metrics['tp'],
                    metrics['fp'],
                    metrics['fn']
                ])
                
        print(f"CSV 파일 저장 완료: {csv_path}")
    except Exception as e:
        print(f"결과 파일 저장 오류: {str(e)}")
    
    return results
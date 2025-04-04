import random
import numpy as np
import torch
import os
import shutil
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def set_seed(seed=42):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def save_confusion_matrix_png(cm, class_names, save_path):
    """혼동 행렬을 PNG로 저장하는 함수"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names,
               yticklabels=class_names,
               annot_kws={'size': 16})
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Class', fontsize=14, labelpad=10)
    plt.xlabel('Predicted Class', fontsize=14, labelpad=10)
    plt.xticks(fontsize=12, rotation=0, ha='right')
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def calculate_and_save_f1_overlap(test_loader, models, device, save_dir):
    """F1 Overlap Score를 계산하고 저장하는 함수"""
    print("F1 Overlap Score 계산 시작...")
    
    # 활동 세그먼트 추출용 함수
    def extract_segments(labels):
        if len(labels) == 0:
            return []
        
        segments = []
        current_class = labels[0]
        start_frame = 0
        
        for i in range(1, len(labels)):
            if labels[i] != current_class:
                segments.append([int(start_frame), int(i-1), int(current_class)])
                current_class = labels[i]
                start_frame = i
        
        segments.append([int(start_frame), int(len(labels)-1), int(current_class)])
        return segments
    
    all_true_segments = []
    all_pred_segments = []
    
    # 모델을 평가 모드로 설정
    for model in models:
        model.eval()
    
    # 모든 배치 처리
    with torch.no_grad():
        for batch in test_loader:
            coords, labels, _ = batch
            coords = coords.to(device, dtype=torch.float32)
            
            # 모든 모델의 예측 수집
            batch_preds = []
            for model in models:
                outputs = model(coords, conservative_no_activity=True, apply_transition_rules=True)
                
                if isinstance(outputs, tuple):
                    _, predictions = outputs
                else:
                    _, predictions = torch.max(outputs, dim=-1)
                
                batch_preds.append(predictions)
            
            # 앙상블 (다수결)
            batch_preds = torch.stack(batch_preds)
            ensemble_preds = torch.mode(batch_preds, dim=0).values
            
            # 각 배치 항목에 대해 세그먼트 추출
            for i in range(coords.shape[0]):
                if labels is not None:
                    true_labels = labels[i].cpu().numpy()
                    pred_labels = ensemble_preds[i].cpu().numpy()
                    
                    true_segs = extract_segments(true_labels)
                    pred_segs = extract_segments(pred_labels)
                    
                    all_true_segments.extend(true_segs)
                    all_pred_segments.extend(pred_segs)
    
    print(f"추출된 세그먼트: 실제={len(all_true_segments)}개, 예측={len(all_pred_segments)}개")
    
    # Overlap F1 Score 계산
    def calculate_f1(true_segments, pred_segments, thresholds=[0.25, 0.5]):
        results = {}
        
        for threshold in thresholds:
            tp = 0
            matched_preds = set()
            
            for t_start, t_end, t_class in true_segments:
                t_duration = t_end - t_start + 1
                best_overlap = 0
                best_pred_idx = None
                
                for p_idx, (p_start, p_end, p_class) in enumerate(pred_segments):
                    if p_idx in matched_preds or p_class != t_class:
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
            
            fp = len(pred_segments) - len(matched_preds)
            fn = len(true_segments) - tp
            
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
    
    # 파일 저장 디렉토리 확인
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # F1 계산
        thresholds = [0.25, 0.5]
        f1_results = calculate_f1(all_true_segments, all_pred_segments, thresholds)
        
        # 텍스트 파일 저장
        txt_path = os.path.join(save_dir, "overlap_f1_results.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("F1 Overlap Score 결과\n\n")
            f.write(f"총 세그먼트 수: 실제={len(all_true_segments)}, 예측={len(all_pred_segments)}\n\n")
            
            for threshold, metrics in f1_results.items():
                f.write(f"임계값 {threshold*100:.0f}%:\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
                f.write(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}\n\n")
        
        print(f"F1 Overlap Score 텍스트 결과 저장 완료: {txt_path}")
        
        # CSV 파일 저장
        csv_path = os.path.join(save_dir, "overlap_f1_results.csv")
        with open(csv_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(['Threshold', 'Precision', 'Recall', 'F1', 'TP', 'FP', 'FN'])
            for threshold, metrics in f1_results.items():
                writer.writerow([
                    f"{threshold*100:.0f}%", 
                    f"{metrics['precision']:.4f}", 
                    f"{metrics['recall']:.4f}", 
                    f"{metrics['f1']:.4f}",
                    metrics['tp'],
                    metrics['fp'],
                    metrics['fn']
                ])
        
        print(f"F1 Overlap Score CSV 결과 저장 완료: {csv_path}")
        
        # 차트 저장
        plt.figure(figsize=(10, 6))
        thresholds_list = list(f1_results.keys())
        f1_scores = [f1_results[t]['f1'] for t in thresholds_list]
        
        plt.bar(range(len(thresholds_list)), f1_scores, color='skyblue')
        plt.xticks(range(len(thresholds_list)), [f"{t*100:.0f}%" for t in thresholds_list])
        plt.ylabel('F1 Score')
        plt.title('F1 Overlap Score by Threshold')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, score in enumerate(f1_scores):
            plt.text(i, score + 0.02, f"{score:.4f}", ha='center', va='bottom')
        
        plt.tight_layout()
        chart_path = os.path.join(save_dir, "overlap_f1_chart.png")
        plt.savefig(chart_path, dpi=100)
        plt.close()
        
        print(f"F1 Overlap Score 차트 저장 완료: {chart_path}")
        
        return f1_results
    except Exception as e:
        print(f"F1 Overlap Score 계산/저장 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return None 
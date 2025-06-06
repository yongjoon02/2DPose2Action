import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc

# 상위 디렉토리를 path에 추가하여 src 패키지를 인식하도록 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import TCN
from src.yolo_pose_csv_kdy import process_video, process_csv_files, get_available_folder
from src.datasets import class_mapping
from src.utils import set_seed
from src.Rerun_visualize import rerun_visualize

def load_model_from_checkpoint(checkpoint_path, config, device):
    """체크포인트에서 모델을 로드하는 함수"""
    # use_se 매개변수가 없는 경우 기본값 False 사용
    use_se = config["model"].get("use_se", False)
    
    model = TCN(
        input_size=config["model"]["input_size"],
        output_size=config["model"]["output_size"],
        num_channels=config["model"]["hidden_channels"],
        kernel_size=config["model"]["kernel_size"],
        dropout=config["model"]["dropout"],
        use_se=use_se
    ).to(device)
    
    # 체크포인트에서 가중치 로드 (strict=False로 설정하여 일부 키 불일치 허용)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model.eval()
    print("모델 로드 완료 (일부 가중치 불일치가 있을 수 있으나 무시됨)")
    return model

def preprocess_csv(csv_path):
    """CSV 파일을 모델 입력 형식으로 전처리하는 함수"""
    # CSV 파일 로드
    df = pd.read_csv(csv_path)
    
    # joint 좌표 추출 (person_detected와 신뢰도 정보 제외)
    coords = []
    for i in range(1, 18):  # 17개 관절
        x = df[f'joint{i}_x'].values
        y = df[f'joint{i}_y'].values
        coords.extend([x, y])
    
    # (시퀀스 길이, 특징 수)로 변환
    coords = np.array(coords).T
    
    # 이동 평균 적용
    window_size = 3
    smoothed = np.copy(coords)
    for i in range(coords.shape[1]):
        smoothed[:, i] = np.convolve(coords[:, i], 
                                   np.ones(window_size)/window_size, 
                                   mode='same')
    
    # 정규화
    mean = np.mean(smoothed, axis=0)
    std = np.std(smoothed, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (smoothed - mean) / std
    
    return torch.FloatTensor(normalized)

def process_predictions(pred_sequence):
    """예측 시퀀스를 JSON 형식으로 변환"""
    class_indices = {i: name for i, name in enumerate(class_mapping.values())}
    
    if len(pred_sequence) == 0:
        return []
    
    current_class = pred_sequence[0]
    start_frame = 0
    result = []
    
    for frame, pred in enumerate(pred_sequence):
        if pred != current_class:
            result.append({
                "frameRange": [int(start_frame), int(frame - 1)],
                "activity": class_indices[int(current_class)]
            })
            current_class = pred
            start_frame = frame
    
    result.append({
        "frameRange": [int(start_frame), int(len(pred_sequence) - 1)],
        "activity": class_indices[int(current_class)]
    })
    
    return result

# 기존 process_video, process_csv_files 함수를 수정하여 person_detected와 신뢰도 정보를 제외하는 함수
def process_csv_files_without_confidence(src_folder, dest_folder):
    """CSV 파일을 후처리하고 person_detected 및 신뢰도 정보를 제외하는 함수"""
    # 결과 저장 폴더가 없으면 생성
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # src_folder 내의 모든 CSV 파일에 대해 처리
    for filename in os.listdir(src_folder):
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(src_folder, filename)
            df = pd.read_csv(file_path)
            
            # 결과를 저장할 DataFrame 생성 (프레임 번호 추가)
            result = pd.DataFrame()
            result['frame'] = df['frame']
            
            # 각 관절의 x, y 좌표만 추출 (confidence 제외)
            joint_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                           'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                           'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                           'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
            
            for i, name in enumerate(joint_names):
                result[f'joint{i+1}_x'] = df[f'{name}_x']
                result[f'joint{i+1}_y'] = df[f'{name}_y']
                # confidence 정보는 제외
            
            # 결과를 저장할 경로 지정 (파일명은 원본과 동일)
            save_path = os.path.join(dest_folder, filename)
            result.to_csv(save_path, index=False)
            print(f"저장 완료: {save_path}")

def merge_single_frame_predictions(predictions):
    """
    단일 프레임 예측을 앞뒤 프레임의 활동으로 병합합니다.
    
    Args:
        predictions: 활동 예측 리스트 (각 항목은 frameRange와 activity를 포함)
    
    Returns:
        병합된 예측 리스트
    """
    if len(predictions) <= 1:
        return predictions
    
    merged_predictions = []
    i = 0
    
    while i < len(predictions):
        current = predictions[i]
        
        # 단일 프레임 예측인지 확인 (시작과 끝 프레임이 같은 경우)
        is_single_frame = current["frameRange"][0] == current["frameRange"][1]
        
        if not is_single_frame:
            # 단일 프레임이 아니면 그대로 추가
            merged_predictions.append(current)
            i += 1
            continue
        
        # 단일 프레임 예측인 경우 처리
        prev_activity = predictions[i-1]["activity"] if i > 0 else None
        next_activity = predictions[i+1]["activity"] if i < len(predictions)-1 else None
        
        # 앞뒤 활동이 같으면 그 활동으로 대체
        if prev_activity and next_activity and prev_activity == next_activity:
            # 이전 프레임 범위를 확장
            if len(merged_predictions) > 0:
                merged_predictions[-1]["frameRange"][1] = current["frameRange"][1]
            else:
                # 첫 번째 항목인 경우 (이론적으로 가능성 낮음)
                merged_predictions.append(current)
        # 앞 활동으로 병합
        elif prev_activity:
            if len(merged_predictions) > 0:
                merged_predictions[-1]["frameRange"][1] = current["frameRange"][1]
            else:
                merged_predictions.append(current)
                merged_predictions[-1]["activity"] = prev_activity
        # 뒤 활동으로 병합
        elif next_activity:
            merged_predictions.append({
                "frameRange": current["frameRange"],
                "activity": next_activity
            })
        else:
            # 앞뒤가 없는 경우 (단일 항목인 경우)
            merged_predictions.append(current)
        
        i += 1
    
    # 연속된 같은 활동 병합
    final_predictions = []
    for pred in merged_predictions:
        if not final_predictions or final_predictions[-1]["activity"] != pred["activity"]:
            final_predictions.append(pred)
        else:
            final_predictions[-1]["frameRange"][1] = pred["frameRange"][1]
    
    return final_predictions

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

def calculate_ap(y_true, y_scores, class_idx):
    """단일 클래스에 대한 Average Precision 계산"""
    y_true_binary = (y_true == class_idx).astype(int)
    y_scores_class = y_scores[:, class_idx]
    
    precision, recall, _ = precision_recall_curve(y_true_binary, y_scores_class)
    ap = auc(recall, precision)
    return ap

def calculate_ap_and_map(y_true, y_scores):
    """클래스별 AP와 mAP를 계산하는 함수"""
    num_classes = y_scores.shape[1]
    ap_scores = {}
    
    for class_idx in range(num_classes):
        ap = calculate_ap(y_true, y_scores, class_idx)
        ap_scores[class_idx] = ap
    
    mAP = np.mean(list(ap_scores.values()))
    return ap_scores, mAP

def calculate_overlap_f1(true_activities, pred_activities, overlap_thresholds=[0.25, 0.5]):
    """시간 구간 기반의 F1 점수를 계산"""
    results = {}
    
    for threshold in overlap_thresholds:
        tp = 0
        fp = 0
        fn = 0
        matched_preds = set()
        
        for t_start, t_end, t_class in true_activities:
            t_duration = t_end - t_start + 1
            best_overlap = 0
            best_pred_idx = None
            
            for i, (p_start, p_end, p_class) in enumerate(pred_activities):
                if i in matched_preds or t_class != p_class:
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
                        best_pred_idx = i
            
            if best_overlap >= threshold:
                tp += 1
                matched_preds.add(best_pred_idx)
            else:
                fn += 1
        
        fp = len(pred_activities) - len(matched_preds)
        
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
    """연속 프레임 레이블을 활동 구간 리스트로 변환"""
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
    
    segments.append([start_frame, frames[-1], current_class])
    return segments

def numpy_to_python(obj):
    """NumPy 데이터 타입을 기본 Python 타입으로 변환"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    else:
        return obj

def calculate_segmental_edit_score(true_labels, pred_labels):
    """Segmental Edit Score 계산"""
    def get_segments(labels):
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
    
    true_segments = get_segments(true_labels)
    pred_segments = get_segments(pred_labels)
    
    true_segment_labels = [seg[2] for seg in true_segments]
    pred_segment_labels = [seg[2] for seg in pred_segments]
    
    edit_distance = levenshtein_distance(true_segment_labels, pred_segment_labels)
    
    max_distance = max(len(true_segment_labels), len(pred_segment_labels))
    if max_distance == 0:
        edit_score = 100.0
    else:
        edit_score = (1 - edit_distance / max_distance) * 100
    
    segments_info = {
        "true_segments": true_segments,
        "pred_segments": pred_segments,
        "num_true_segments": len(true_segments),
        "num_pred_segments": len(pred_segments),
        "edit_distance": edit_distance,
        "normalized_score": edit_score
    }
    
    return edit_score, segments_info

def convert_json_labels_to_sequence(json_labels, sequence_length):
    """JSON 형식의 라벨을 프레임별 시퀀스로 변환하는 함수"""
    # 클래스 이름을 인덱스로 변환하는 매핑 생성
    class_to_idx = {name: idx for idx, name in enumerate(class_mapping.values())}
    
    # 초기값으로 전체 시퀀스를 -1로 채움 (유효하지 않은 라벨)
    sequence = np.ones(sequence_length, dtype=int) * -1
    
    # JSON 라벨의 각 활동 구간에 대해
    for activity in json_labels:
        start_frame = activity["frameRange"][0]
        end_frame = activity["frameRange"][1]
        activity_name = activity["activity"]
        
        # 활동 이름이 매핑에 있는지 확인
        if activity_name in class_to_idx:
            activity_idx = class_to_idx[activity_name]
            
            # 프레임 범위가 유효한지 확인
            start_frame = max(0, start_frame)
            end_frame = min(sequence_length - 1, end_frame)
            
            # 해당 범위에 활동 인덱스 할당
            if start_frame <= end_frame:
                sequence[start_frame:end_frame+1] = activity_idx
    
    # 라벨이 없는 프레임이 있는지 확인하고 처리
    if np.any(sequence == -1):
        print(f"경고: 일부 프레임({np.sum(sequence == -1)}개)에 라벨이 없습니다. 이 프레임은 평가에서 제외됩니다.")
        # 라벨이 없는 프레임 마스킹 위해 유효한 인덱스만 반환
        valid_indices = np.where(sequence != -1)[0]
        return sequence, valid_indices
    
    return sequence, np.arange(sequence_length)

def inference_video(video_path, model, device, temp_dir, output_dir, label_dir=None):
    """비디오에 대한 추론을 수행하는 함수"""
    # YOLO 모델 경로
    yolo_model_path = "yolo.pt"
    
    # 비디오 이름 추출
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 결과 디렉토리 생성
    os.makedirs(temp_dir, exist_ok=True)
    
    # 각 비디오별 결과 디렉토리 생성
    video_result_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_result_dir, exist_ok=True)
    
    # 중간 처리 디렉토리 생성
    csv_dir = os.path.join(video_result_dir, "csv")
    processed_dir = os.path.join(video_result_dir, "processed")
    results_dir = os.path.join(video_result_dir, "results")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. YOLO를 사용하여 포즈 추정 및 CSV 생성
    print("포즈 추정 중...")
    process_video(video_path, yolo_model_path, device, csv_dir)
    
    # 2. CSV 파일 후처리 (person_detected 및 신뢰도 정보 제외)
    print("CSV 파일 후처리 중...")
    process_csv_files_without_confidence(csv_dir, processed_dir)
    
    # 3. 각 CSV 파일에 대해 추론 수행
    print("활동 분류 추론 중...")
    all_json_results = []
    all_labels = []
    all_preds = []
    all_scores = []
    total_frames = 0
    
    # CSV 파일별 예측 결과를 저장하기 위한 딕셔너리
    file_predictions = {}
    
    # 정답 라벨 로드 (있는 경우)
    if label_dir:
        # 기존 방식: video_name_label.json 형식 검색
        label_file = os.path.join(label_dir, f"{video_name}_label.json")
        
        # 새로운 방식: video_name과 비슷한 이름의 json 파일 찾기
        if not os.path.exists(label_file):
            # 디렉토리에서 모든 json 파일 검색
            json_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
            
            # 비디오 이름과 가장 유사한 json 파일 찾기
            for json_file in json_files:
                if video_name in json_file or json_file.split('_')[0] in video_name:
                    label_file = os.path.join(label_dir, json_file)
                    print(f"비디오 {video_name}에 대응하는 라벨 파일 찾음: {json_file}")
                    break
        
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                true_labels = json.load(f)
            print(f"정답 라벨 로드 완료: {label_file}")
        else:
            print(f"경고: 정답 라벨 파일을 찾을 수 없습니다: {video_name}")
            true_labels = None
    else:
        true_labels = None
    
    # 먼저 총 프레임 수 계산 (정답 라벨 변환을 위해)
    for csv_file in os.listdir(processed_dir):
        if not csv_file.endswith('.csv'):
            continue
        
        csv_path = os.path.join(processed_dir, csv_file)
        df = pd.read_csv(csv_path)
        total_frames += len(df)
    
    # 정답 라벨을 시퀀스로 변환
    if true_labels:
        print(f"정답 라벨을 시퀀스로 변환 중 (총 {total_frames}개 프레임)...")
        label_sequence, valid_indices = convert_json_labels_to_sequence(true_labels, total_frames)
    else:
        label_sequence = None
        valid_indices = None
    
    processed_frames = 0
    for csv_file in os.listdir(processed_dir):
        if not csv_file.endswith('.csv'):
            continue
        
        csv_path = os.path.join(processed_dir, csv_file)
        
        # CSV 파일 전처리
        inputs = preprocess_csv(csv_path)
        inputs = inputs.unsqueeze(0).to(device)
        sequence_length = inputs.shape[1]
        
        # 추론 수행
        with torch.no_grad():
            outputs = model(inputs, conservative_no_activity=True, apply_transition_rules=True)
            if isinstance(outputs, tuple):
                logits, predictions = outputs
            else:
                logits = outputs
                _, predictions = torch.max(logits, dim=-1)
            
            # 확률 점수 계산
            scores = torch.nn.functional.softmax(logits, dim=-1)
            
            # 가중치 적용 없이 원래 예측 그대로 사용
            # 모델이 훈련된 대로 추론 결과를 사용하는 것이 더 정확함
        
        # 예측 결과를 JSON으로 변환
        pred_sequence = predictions[0].cpu().numpy()
        json_result = process_predictions(pred_sequence)
        
        # 결과 저장
        output_json = os.path.join(results_dir, csv_file.replace('.csv', '_prediction.json'))
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=4, ensure_ascii=False)
        
        # CSV 파일별 상세 예측 결과 저장 - get_filename과 비슷한 형식으로 맞춤
        class_indices = {i: name for i, name in enumerate(class_mapping.values())}
        
        # 파일 예측 정보 생성
        file_prediction = {
            "frames": list(range(sequence_length)),
            "predicted_labels": pred_sequence.tolist(),
            "predicted_activities": [class_indices[int(label)] for label in pred_sequence],
            "probabilities": scores[0].cpu().numpy().tolist(),
            "activity_segments": json_result
        }
        
        # 정답 라벨이 있는 경우 추가
        if label_sequence is not None:
            current_frame_labels = label_sequence[processed_frames:processed_frames+sequence_length]
            if valid_indices is not None:
                # 유효한 인덱스만 필터링
                valid_mask = np.where((valid_indices >= processed_frames) & 
                                    (valid_indices < processed_frames+sequence_length))[0]
                valid_local_indices = valid_indices[valid_mask] - processed_frames
                
                if len(valid_local_indices) > 0:
                    file_prediction["true_labels"] = current_frame_labels[valid_local_indices].tolist()
                    file_prediction["true_activities"] = [
                        class_indices[int(label)] if int(label) in class_indices else f"unknown_{label}" 
                        for label in file_prediction["true_labels"]
                    ]
            else:
                file_prediction["true_labels"] = current_frame_labels.tolist()
                file_prediction["true_activities"] = [
                    class_indices[int(label)] if int(label) in class_indices else f"unknown_{label}" 
                    for label in file_prediction["true_labels"]
                ]
        
        # 파일 이름에서 비유효 문자 제거
        safe_file_name = os.path.basename(csv_file).replace(".csv", "")
        safe_file_name = ''.join(c for c in safe_file_name if c.isalnum() or c in '._- ')
        
        # CSV 파일별 예측 결과 저장
        file_prediction_path = os.path.join(results_dir, f"{safe_file_name}_detailed_prediction.json")
        with open(file_prediction_path, 'w', encoding='utf-8') as f:
            json.dump(file_prediction, f, indent=4, ensure_ascii=False)
        
        # 전체 결과 리스트에 저장
        all_json_results.extend(json_result)
        all_preds.extend(pred_sequence.reshape(-1))
        all_scores.append(scores[0].cpu().numpy().reshape(-1, scores.shape[-1]))
        
        # 정답 라벨이 있는 경우, 현재 CSV에 해당하는 프레임의 라벨 추출
        if label_sequence is not None:
            current_frame_labels = label_sequence[processed_frames:processed_frames+sequence_length]
            # 이 부분의 유효한 인덱스만 필터링
            if valid_indices is not None:
                valid_mask = np.where((valid_indices >= processed_frames) & 
                                     (valid_indices < processed_frames+sequence_length))[0]
                valid_local_indices = valid_indices[valid_mask] - processed_frames
                if len(valid_local_indices) > 0:
                    # 유효한 라벨만 추가
                    all_labels.extend(current_frame_labels[valid_local_indices])
            else:
                all_labels.extend(current_frame_labels)
        
        # 파일 예측 결과 딕셔너리에 추가
        file_predictions[safe_file_name] = file_prediction
        
        processed_frames += sequence_length
        print(f"처리 완료: {csv_file}")
    
    # 4. 단일 프레임 예측 병합하여 최종 JSON 파일 생성
    print("단일 프레임 예측 병합 중...")
    merged_predictions = merge_single_frame_predictions(all_json_results)
    
    # 5. 최종 결과를 JSON 파일로 저장
    final_result_file = os.path.join(results_dir, f"{video_name}_prediction.json")
    with open(final_result_file, "w", encoding='utf-8') as f:
        json.dump(merged_predictions, f, indent=4, ensure_ascii=False)
    
    # 모든 파일의 예측 결과를 하나의 JSON으로 저장
    all_file_predictions_path = os.path.join(results_dir, f"{video_name}_all_file_predictions.json")
    with open(all_file_predictions_path, 'w', encoding='utf-8') as f:
        json.dump(file_predictions, f, indent=4, ensure_ascii=False)
    
    # 6. 평가 지표 계산 및 저장
    if all_preds:
        all_preds = np.array(all_preds)
        all_scores = np.vstack(all_scores) if all_scores else np.array([])
        
        # 추후 종합 평가를 위해 예측값과 점수 저장
        np.save(os.path.join(results_dir, "predictions.npy"), all_preds)
        np.save(os.path.join(results_dir, "scores.npy"), all_scores)
        
        if len(all_labels) > 0:
            all_labels = np.array(all_labels)
            np.save(os.path.join(results_dir, "labels.npy"), all_labels)
        
        # 클래스 이름 설정
        class_names = list(class_mapping.values())
        
        # 개별 비디오에 대한 평가 지표는 결과 파일만 저장하고 종합 평가는 하지 않음
        # 세그먼트 예측 결과만 개별적으로 저장
    
    print(f"예측 결과가 저장되었습니다: {final_result_file}")
    return final_result_file

def main():
    # 기본 설정
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 장치: {device}")
    
    try:
        # 설정 파일 로드
        config_path = r"F:\yolo\result\va_tcn_result8\combined_results\hyperparameters.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 모델 체크포인트 로드
        checkpoint_path = r"F:\yolo\result\va_tcn_result8\best_model.pth"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"모델 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        
        model = load_model_from_checkpoint(checkpoint_path, config, device)
        
        # 비디오 파일 경로 설정
        video_dir = "data/inference_video"
        # 정답 라벨 디렉토리 설정
        label_dir = "data/inference_labels"  # 정답 라벨이 있는 디렉토리
        
        if not os.path.exists(video_dir):
            raise FileNotFoundError(f"비디오 디렉토리를 찾을 수 없습니다: {video_dir}")
        
        if not os.path.exists(label_dir):
            print(f"경고: 정답 라벨 디렉토리를 찾을 수 없습니다: {label_dir}")
            print("평가 지표 계산 없이 추론만 진행합니다.")
            label_dir = None
        
        # 임시 디렉토리와 출력 디렉토리 설정
        temp_dir = "temp_inference"
        output_dir = "inference_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # 비디오 디렉토리 내의 모든 mp4 파일에 대해 처리
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        if not video_files:
            print(f"경고: {video_dir}에 MP4 파일이 없습니다.")
            return
        
        # 종합 결과를 저장할 디렉토리
        summary_dir = os.path.join(output_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        
        # 종합 평가를 위한 변수들
        all_combined_preds = []
        all_combined_labels = []
        all_combined_scores = []
        
        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)
            print(f"\n비디오 처리 중: {video_file}")
            result_file = inference_video(video_path, model, device, temp_dir, output_dir, label_dir)
            
            # 평가 데이터 수집 (있는 경우)
            video_name = os.path.splitext(video_file)[0]
            video_result_dir = os.path.join(output_dir, video_name)
            results_dir = os.path.join(video_result_dir, "results")
            preds_file = os.path.join(results_dir, "predictions.npy")
            labels_file = os.path.join(results_dir, "labels.npy")
            scores_file = os.path.join(results_dir, "scores.npy")
            
            if os.path.exists(preds_file) and os.path.exists(scores_file):
                preds = np.load(preds_file)
                all_combined_preds.extend(preds)
                
                if os.path.exists(labels_file):
                    labels = np.load(labels_file)
                    all_combined_labels.extend(labels)
                
                scores = np.load(scores_file)
                all_combined_scores.append(scores)
        
        # 종합 평가 수행 (데이터가 있는 경우)
        if len(all_combined_preds) > 0 and len(all_combined_scores) > 0:
            all_combined_preds = np.array(all_combined_preds)
            if len(all_combined_scores) > 0:
                all_combined_scores = np.vstack(all_combined_scores)
            
            # 클래스 이름 설정
            class_names = list(class_mapping.values())
            
            # 평가 결과 디렉토리
            evaluation_dir = os.path.join(summary_dir, "evaluation")
            os.makedirs(evaluation_dir, exist_ok=True)
            
            # 혼동 행렬 저장 (라벨이 있는 경우에만)
            if len(all_combined_labels) > 0:
                all_combined_labels = np.array(all_combined_labels)
                
                # 혼동 행렬 계산 및 저장
                cm = confusion_matrix(all_combined_labels, all_combined_preds)
                save_confusion_matrix_png(
                    cm,
                    class_names,
                    os.path.join(evaluation_dir, "confusion_matrix.png")
                )
                
                # 혼동 행렬을 CSV로도 저장
                cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
                cm_df.to_csv(os.path.join(evaluation_dir, "confusion_matrix.csv"))
                
                # 분류 보고서 저장
                report = classification_report(all_combined_labels, all_combined_preds, 
                                            target_names=class_names, output_dict=True)
                
                # 분류 보고서를 텍스트와 JSON으로 저장
                with open(os.path.join(evaluation_dir, "classification_report.txt"), "w") as f:
                    f.write(classification_report(all_combined_labels, all_combined_preds, 
                                               target_names=class_names))
                
                with open(os.path.join(evaluation_dir, "classification_report.json"), "w") as f:
                    json.dump(report, f, indent=4)
                
                # 클래스별 정확도 계산 및 저장
                class_accuracy = {}
                for i, class_name in enumerate(class_names):
                    mask = (all_combined_labels == i)
                    if np.sum(mask) > 0:  # 해당 클래스의 샘플이 있는 경우만
                        acc = np.mean(all_combined_preds[mask] == i) * 100
                        class_accuracy[class_name] = acc
                
                with open(os.path.join(evaluation_dir, "class_accuracy.json"), "w") as f:
                    json.dump(class_accuracy, f, indent=4)
                
                # Segmental Edit Score 계산 및 저장
                edit_score, segments_info = calculate_segmental_edit_score(
                    all_combined_labels, all_combined_preds)
                
                with open(os.path.join(evaluation_dir, "segmental_edit_score.json"), "w") as f:
                    json.dump({
                        "edit_score": float(edit_score),
                        "num_true_segments": int(segments_info["num_true_segments"]),
                        "num_pred_segments": int(segments_info["num_pred_segments"]),
                        "edit_distance": int(segments_info["edit_distance"])
                    }, f, indent=4)
                
                # AP와 mAP 계산 및 저장
                ap_scores, mAP = calculate_ap_and_map(all_combined_labels, all_combined_scores)
                ap_results = {
                    "class_ap": {class_names[i]: float(ap) for i, ap in ap_scores.items()},
                    "mAP": float(mAP)
                }
                
                with open(os.path.join(evaluation_dir, "average_precision.json"), "w") as f:
                    json.dump(ap_results, f, indent=4)
                
                # Overlap F1 Score 계산 및 저장
                true_segments = extract_activity_segments(all_combined_labels)
                pred_segments = extract_activity_segments(all_combined_preds)
                overlap_results = calculate_overlap_f1(true_segments, pred_segments)
                
                with open(os.path.join(evaluation_dir, "overlap_f1_scores.json"), "w") as f:
                    # NumPy 값을 Python 기본 타입으로 변환
                    json.dump(numpy_to_python(overlap_results), f, indent=4)
                
                print(f"\n종합 평가 결과가 {evaluation_dir}에 저장되었습니다.")
                print(f"전체 정확도: {np.mean(all_combined_preds == all_combined_labels)*100:.2f}%")
                print(f"mAP: {mAP*100:.2f}%")
            else:
                print("\n경고: 정답 라벨이 없어 종합 평가를 수행할 수 없습니다.")
        
        print(f"\n추론 완료! 결과는 {output_dir} 디렉토리에 저장되었습니다.")
        print(f"종합 결과는 {summary_dir}에 저장되었습니다.")
        
        print("\n시각화 시작...")
        try:
            for video_file in video_files:
                video_name = os.path.splitext(video_file)[0]
                rerun_visualize(
                    csv_dir=Path(os.path.join(output_dir, video_name, "processed")),
                    pred_dir=Path(os.path.join(output_dir, video_name, "results")),
                    video_dir=Path(video_dir),
                    title=f"skeleton_label_compare_{video_name}"
                )
        except Exception as e:
            print(f"시각화 중 오류 발생: {e}")
            print("시각화는 실패했지만 예측 결과는 정상적으로 저장되었습니다.")
        
    except FileNotFoundError as e:
        print(f"오류: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON 파일 파싱 오류: {e}")
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
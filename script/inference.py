import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# 상위 디렉토리를 path에 추가하여 src 패키지를 인식하도록 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import TCN
from src.yolo_pose_csv_kdy import process_video, process_csv_files, get_available_folder
from src.datasets import class_mapping
from src.utils import set_seed
from src.Rerun_visualize import rerun_visualize

def load_model_from_checkpoint(checkpoint_path, config, device):
    """체크포인트에서 모델을 로드하는 함수"""
    model = TCN(
        input_size=config["model"]["input_size"],
        output_size=config["model"]["output_size"],
        num_channels=config["model"]["hidden_channels"],
        kernel_size=config["model"]["kernel_size"],
        dropout=config["model"]["dropout"],
        use_se=config["model"]["use_se"]
    ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
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

def inference_video(video_path, model, device, temp_dir, output_dir):
    """비디오에 대한 추론을 수행하는 함수"""
    # YOLO 모델 경로
    yolo_model_path = "yolo.pt"  # YOLO 모델 파일 경로 지정 필요
    
    # 결과 디렉토리 생성
    os.makedirs(temp_dir, exist_ok=True)
    
    # 각 비디오별 결과 디렉토리 생성
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_result_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_result_dir, exist_ok=True)
    
    # 중간 처리 디렉토리 생성
    csv_dir = os.path.join(video_result_dir, "csv")
    processed_dir = os.path.join(video_result_dir, "processed")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. YOLO를 사용하여 포즈 추정 및 CSV 생성
    print("포즈 추정 중...")
    process_video(video_path, yolo_model_path, device, csv_dir)
    
    # 2. CSV 파일 후처리 (person_detected 및 신뢰도 정보 제외)
    print("CSV 파일 후처리 중...")
    process_csv_files_without_confidence(csv_dir, processed_dir)
    
    # 3. 각 CSV 파일에 대해 추론 수행
    print("활동 분류 추론 중...")
    all_json_results = []
    results_dir = os.path.join(video_result_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    for csv_file in os.listdir(processed_dir):
        if not csv_file.endswith('.csv'):
            continue
        
        csv_path = os.path.join(processed_dir, csv_file)
        
        # CSV 파일 전처리
        inputs = preprocess_csv(csv_path)
        inputs = inputs.unsqueeze(0).to(device)  # 배치 차원 추가
        
        # 추론 수행
        with torch.no_grad():
            outputs = model(inputs, conservative_no_activity=True, apply_transition_rules=True)
            if isinstance(outputs, tuple):
                predictions = outputs[1]
            else:
                _, predictions = torch.max(outputs, dim=-1)
        
        # 예측 결과를 JSON으로 변환
        pred_sequence = predictions[0].cpu().numpy()
        json_result = process_predictions(pred_sequence)
        
        # 결과 저장
        output_json = os.path.join(results_dir, csv_file.replace('.csv', '_prediction.json'))
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=4, ensure_ascii=False)
        
        # 전체 결과 리스트에 저장 (수정: 누적되도록 변경)
        all_json_results.extend(json_result)
        print(f"처리 완료: {csv_file}")

    # 4. 단일 프레임 예측 병합하여 최종 JSON 파일 생성
    print("단일 프레임 예측 병합 중...")
    merged_predictions = merge_single_frame_predictions(all_json_results)
    
    # 5. 최종 결과를 JSON 파일로 저장
    final_result_file = os.path.join(results_dir, f"{video_name}_prediction.json")
    with open(final_result_file, "w", encoding='utf-8') as f:
        json.dump(merged_predictions, f, indent=4, ensure_ascii=False)
    
    print(f"예측 결과가 저장되었습니다: {final_result_file}")
    return final_result_file

def main():
    # 기본 설정
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 장치: {device}")
    
    try:
        # 설정 파일 로드
        config_path = r"result\tcn_result72\combined_results\hyperparameters.json"
        with open(config_path, 'r', encoding='utf-8') as f:  # UTF-8 인코딩 명시
            config = json.load(f)
        
        # 모델 체크포인트 로드
        checkpoint_path = r"result\tcn_result72\combined_results\best_model.pth"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"모델 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        
        model = load_model_from_checkpoint(checkpoint_path, config, device)
        
        # 비디오 파일 경로 설정
        video_dir = "data/inference_video"  # 추론할 비디오가 있는 디렉토리
        if not os.path.exists(video_dir):
            raise FileNotFoundError(f"비디오 디렉토리를 찾을 수 없습니다: {video_dir}")
        
        # 임시 디렉토리와 출력 디렉토리 설정
        temp_dir = "temp_inference"
        output_dir = "inference_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # 비디오 디렉토리 내의 모든 mp4 파일에 대해 처리
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        if not video_files:
            print(f"경고: {video_dir}에 MP4 파일이 없습니다.")
            return
        
        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)
            print(f"\n비디오 처리 중: {video_file}")
            inference_video(video_path, model, device, temp_dir, output_dir)
        
        print(f"\n추론 완료! 결과는 {output_dir} 디렉토리에 저장되었습니다.")
        
        print("\n시각화 시작...")
        rerun_visualize(
            csv_dir=Path(os.path.join(temp_dir, "processed")),
            pred_dir=Path(output_dir),
            video_dir=Path(video_dir),
            title="skeleton_label_compare"
        )
        
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
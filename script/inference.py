import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 상위 디렉토리를 path에 추가하여 src 패키지를 인식하도록 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import TCN
from src.yolo_pose_csv_kdy import process_video, process_csv_files, get_available_folder
from src.datasets import class_mapping
from src.utils import set_seed

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
    import pandas as pd
    
    # CSV 파일 로드
    df = pd.read_csv(csv_path)
    
    # joint 좌표 추출
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

def inference_video(video_path, model, device, temp_dir, output_dir):
    """비디오에 대한 추론을 수행하는 함수"""
    # YOLO 모델 경로
    yolo_model_path = "yolo.pt"  # YOLO 모델 파일 경로 지정 필요
    
    # 임시 디렉토리 생성
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. YOLO를 사용하여 포즈 추정 및 CSV 생성
    print("포즈 추정 중...")
    process_video(video_path, yolo_model_path, device, temp_dir)
    
    # 2. CSV 파일 후처리
    print("CSV 파일 후처리 중...")
    processed_dir = os.path.join(temp_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    process_csv_files(temp_dir, processed_dir)
    
    # 3. 각 CSV 파일에 대해 추론 수행
    print("활동 분류 추론 중...")
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
        output_json = os.path.join(output_dir, csv_file.replace('.csv', '_prediction.json'))
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=4, ensure_ascii=False)
        
        print(f"처리 완료: {csv_file}")

def main():
    # 기본 설정
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 장치: {device}")
    
    try:
        # 설정 파일 로드
        config_path = "result/tcn_result54/combined_results/hyperparameters.json"
        with open(config_path, 'r', encoding='utf-8') as f:  # UTF-8 인코딩 명시
            config = json.load(f)
        
        # 모델 체크포인트 로드
        checkpoint_path = "result/tcn_result54/combined_results/best_model.pth"
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
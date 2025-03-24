#csv 후처리해서 사람검출 여부 배제한거

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import pandas as pd
import numpy as np
import glob
from ultralytics import YOLO
import multiprocessing
import time
import threading
multiprocessing.set_start_method('spawn', force=True)

def process_video(video_path, model_path, device, result_folder):
    # 입력 영상 파일명(확장자 제외) 추출
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 결과 CSV 파일이 저장될 result 폴더 생성 (이미 존재하면 무시)
    os.makedirs(result_folder, exist_ok=True)
    
    print(f"Processing {video_name}: {video_path}")
    
    # YOLO 모델 로드 및 설정
    model = YOLO(model_path)
    model = model.to(device)
    model.conf = 0.2
    model.imgsz = (960, 960)
    
    all_keypoints = []
    
    # 영상 처리 (영상 저장 없이 CSV 데이터만 생성)
    try:
        results = model(video_path,
                        task='pose',
                        stream=True,
                        save=False,      # 영상 저장하지 않음
                        device=device,
                        verbose=False)
    except Exception as e:
        print(f"Error processing video {video_name}: {str(e)}")
        return
    
    # 각 프레임별 키포인트 추출
    for frame_idx, r in enumerate(results):
        print(f"\r{video_name} - Processing frame: {frame_idx + 1}", end="")
        frame_dict = {'frame': frame_idx}
        
        try:
            if (r.keypoints is not None and 
                len(r.keypoints.data) > 0 and 
                r.keypoints.data[0].shape[0] > 0):
                # 첫 번째 감지된 사람의 키포인트 추출
                keypoints = r.keypoints.data[0].cpu().numpy()
                frame_dict['person_detected'] = True
                for i, name in enumerate(['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                                           'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                                           'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                                           'left_knee', 'right_knee', 'left_ankle', 'right_ankle']):
                    if i < len(keypoints):
                        frame_dict[f'{name}_x'] = float(keypoints[i][0])
                        frame_dict[f'{name}_y'] = float(keypoints[i][1])
                        frame_dict[f'{name}_conf'] = float(keypoints[i][2])
                    else:
                        frame_dict[f'{name}_x'] = 0.0
                        frame_dict[f'{name}_y'] = 0.0
                        frame_dict[f'{name}_conf'] = 0.0
            else:
                # 사람이 감지되지 않은 경우
                frame_dict['person_detected'] = False
                for name in ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                             'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                             'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                             'left_knee', 'right_knee', 'left_ankle', 'right_ankle']:
                    frame_dict[f'{name}_x'] = 0.0
                    frame_dict[f'{name}_y'] = 0.0
                    frame_dict[f'{name}_conf'] = 0.0
        except Exception as e:
            print(f"\n{video_name} - Error processing frame {frame_idx}: {str(e)}")
            frame_dict['person_detected'] = False
            for name in ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                         'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                         'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                         'left_knee', 'right_knee', 'left_ankle', 'right_ankle']:
                frame_dict[f'{name}_x'] = 0.0
                frame_dict[f'{name}_y'] = 0.0
                frame_dict[f'{name}_conf'] = 0.0
                
        all_keypoints.append(frame_dict)
    
    # 전체 프레임의 키포인트 데이터를 CSV 파일로 저장 (파일명은 입력 영상명 기준)
    if all_keypoints:
        df = pd.DataFrame(all_keypoints)
        csv_path = os.path.join(result_folder, f"{video_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n{video_name} - Keypoints saved to {csv_path}")
    
    print(f"\n{video_name} - Total frames processed: {len(all_keypoints)}")

def process_csv_files(src_folder, dest_folder):
    """CSV 파일을 후처리하는 함수"""
    # 결과 저장 폴더가 없으면 생성
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # src_folder 내의 모든 CSV 파일에 대해 처리
    for filename in os.listdir(src_folder):
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(src_folder, filename)
            df = pd.read_csv(file_path, header=None)
            
            # 프레임(첫 번째 열)과 사람 검출여부(두 번째 열)을 제외한 나머지를 3개씩 묶어서 관절 정보로 가정
            num_joints = (df.shape[1] - 2) // 3
            
            # 결과를 저장할 DataFrame 생성 (프레임 번호 추가)
            result = pd.DataFrame()
            result['frame'] = df.iloc[:, 0]
            
            # 각 관절의 x, y 좌표 추출하여 새로운 컬럼으로 추가
            for i in range(num_joints):
                x_index = 2 + i * 3  # 관절의 x 좌표 인덱스
                y_index = 3 + i * 3  # 관절의 y 좌표 인덱스
                result[f'joint{i+1}_x'] = df.iloc[:, x_index]
                result[f'joint{i+1}_y'] = df.iloc[:, y_index]
            
            # 결과를 저장할 경로 지정 (파일명은 원본과 동일)
            save_path = os.path.join(dest_folder, filename)
            result.to_csv(save_path, index=False)
            print("저장 완료:", save_path)

def get_available_folder(base_name):
    """
    이미 존재하는 폴더인 경우 번호를 붙여 새 폴더명 생성
    예: result가 있으면 result1, result1도 있으면 result2 등
    """
    folder_name = base_name
    counter = 1
    
    while os.path.exists(folder_name):
        folder_name = f"{base_name}{counter}"
        counter += 1
    
    return folder_name

def main():
    # CUDA 사용 가능 여부 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print("Using device: cuda")
    else:
        print("CUDA is not available, using CPU")
    
    model_path = r"yolo.pt"
    
    # 결과 폴더 이름 생성 (이미 존재하는 경우 번호 부여)
    # 결과 저장을 위한 상위 디렉토리 생성
    results_parent_dir = "result"
    os.makedirs(results_parent_dir, exist_ok=True)
    
    base_result_folder = os.path.join(results_parent_dir, "skeleton_result")
    result_folder = get_available_folder(base_result_folder)
    print(f"결과 폴더: {result_folder}")
    
    # data 디렉토리 내 모든 mp4 파일 목록 생성
    video_files = glob.glob(os.path.join("data/2d video", "*.mp4"))
    if not video_files:
        print("No video files found in the 'data' directory.")
        return
    
    # multiprocessing을 위한 인자 리스트 생성
    args = [(video, model_path, device, result_folder) for video in video_files]
    
    # 사용 가능한 CPU 코어 수나 파일 수 중 작은 값만큼 프로세스 생성
    pool = multiprocessing.Pool(processes=min(len(video_files), multiprocessing.cpu_count()))
    pool.starmap(process_video, args)
    pool.close()
    pool.join()
    
    print("\nAll videos processed. CSV files saved in the result folder.")
    
    # 생성된 CSV 파일 후처리 수행
    base_dest_folder = os.path.join(results_parent_dir, "skeleton_result_prepro")
    dest_folder = get_available_folder(base_dest_folder)
    print(f"\n후처리 시작: CSV 파일 변환 중... (저장 폴더: {dest_folder})")
    process_csv_files(result_folder, dest_folder)
    print(f"모든 처리 완료! 최종 결과는 {dest_folder} 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()

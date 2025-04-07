# 병렬 처리 없이 YOLO 포즈 추정을 수행하는 스크립트

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import pandas as pd
import numpy as np
import glob
from ultralytics import YOLO
import time

def process_video(video_path, model_path, device, result_folder):
    # 입력 영상 파일명(확장자 제외) 추출
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 결과 CSV 파일이 저장될 result 폴더 생성 (이미 존재하면 무시)
    os.makedirs(result_folder, exist_ok=True)
    
    print(f"처리 중: {video_name}: {video_path}")
    
    # YOLO 모델 로드 및 설정
    model = YOLO(model_path)
    model = model.to(device)
    model.conf = 0.5
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
        print(f"영상 처리 중 오류 발생 {video_name}: {str(e)}")
        return
    
    # 각 프레임별 키포인트 추출
    for frame_idx, r in enumerate(results):
        print(f"\r{video_name} - 프레임 처리 중: {frame_idx + 1}", end="")
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
            print(f"\n{video_name} - 프레임 {frame_idx} 처리 중 오류: {str(e)}")
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
        print(f"\n{video_name} - 키포인트 저장 완료: {csv_path}")
    
    print(f"\n{video_name} - 총 처리된 프레임: {len(all_keypoints)}")

def process_csv_files(src_folder, dest_folder):
    """CSV 파일을 후처리하는 함수"""
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
            
            # 각 관절의 x, y 좌표 추출하여 새로운 컬럼으로 추가
            joint_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                           'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                           'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                           'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
            
            for i, name in enumerate(joint_names):
                result[f'joint{i+1}_x'] = df[f'{name}_x']
                result[f'joint{i+1}_y'] = df[f'{name}_y']
            
            # 결과를 저장할 경로 지정 (파일명은 원본과 동일)
            save_path = os.path.join(dest_folder, filename)
            result.to_csv(save_path, index=False)
            print(f"저장 완료: {save_path}")

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
        print("사용 중인 장치: CUDA")
    else:
        print("CUDA를 사용할 수 없습니다. CPU 사용 중")
    
    '''
    경로는 여기만 수정하시면 될듯????
    '''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    # 모델 경로 설정
    model_path = os.path.join(project_root, "yolo.pt")
    # 결과 폴더 이름 생성
    results_parent_dir = os.path.join(project_root, "result")
    os.makedirs(results_parent_dir, exist_ok=True)
    base_result_folder = os.path.join(results_parent_dir, "skeleton_result")
    result_folder = get_available_folder(base_result_folder)
    print(f"결과 폴더: {result_folder}")
    # 비디오 디렉토리 - 고정 경로 사용
    video_dir = os.path.join(project_root, "data", "2d video")
    print(f"비디오 디렉토리: {video_dir}")
    
    # 디렉토리 존재 확인
    if not os.path.exists(video_dir):
        print(f"오류: 비디오 디렉토리가 존재하지 않습니다: {video_dir}")
        return
    
    # 비디오 파일 목록 생성
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    if not video_files:
        print(f"오류: '{video_dir}' 디렉토리에 MP4 파일이 없습니다.")
        # 디렉토리 내용 출력
        print(f"디렉토리 내용:")
        for item in os.listdir(video_dir):
            print(f" - {item}")
        return
    
    print(f"총 {len(video_files)}개의 비디오 파일을 처리합니다:")
    for video_path in video_files:
        print(f" - {os.path.basename(video_path)}")
    
    # 비디오 파일을 순차적으로 처리
    for video_path in video_files:
        print(f"\n비디오 처리 시작: {os.path.basename(video_path)}")
        process_video(video_path, model_path, device, result_folder)
    
    print("\n모든 비디오 처리 완료. CSV 파일들이 결과 폴더에 저장되었습니다.")
    
    # 생성된 CSV 파일 확인
    csv_files = glob.glob(os.path.join(result_folder, "*.csv"))
    if not csv_files:
        print(f"경고: {result_folder}에 CSV 파일이 생성되지 않았습니다.")
        return
    
    print(f"총 {len(csv_files)}개의 CSV 파일이 생성되었습니다.")
    
    # 생성된 CSV 파일 후처리 수행
    base_dest_folder = os.path.join(results_parent_dir, "skeleton_result_prepro")
    dest_folder = get_available_folder(base_dest_folder)
    print(f"\n후처리 시작: CSV 파일 변환 중... (저장 폴더: {dest_folder})")
    process_csv_files(result_folder, dest_folder)
    print(f"모든 처리 완료! 최종 결과는 {dest_folder} 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()
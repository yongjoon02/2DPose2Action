# 필요한 라이브러리들을 임포트합니다
import os                  # 파일/폴더 경로 관리를 위한 라이브러리
import torch              # 딥러닝 연산을 위한 PyTorch
import pandas as pd       # 데이터프레임 처리를 위한 pandas
import numpy as np        # 수치 연산을 위한 numpy
import cv2               # 영상 처리를 위한 OpenCV
import time              # 시간 지연 기능을 위한 time
import threading         # 멀티스레딩을 위한 threading
import glob              # 파일 경로 패턴 매칭을 위한 glob
from ultralytics import YOLO

def get_next_result_folder():
    # 'F:/yolo/result*' 패턴의 모든 폴더를 찾습니다
    existing_folders = glob.glob("F:/yolo/result*")
    # 기존 폴더가 없으면 기본 'result' 폴더를 반환합니다s
    if not existing_folders:
        return "F:/yolo/result"
    max_num = 0
    for folder in existing_folders:
        # 기본 'result' 폴더가 있으면 최소값을 1로 설정
        if folder == "F:/yolo/result":
            max_num = max(max_num, 1)
        else:
            try:
                # 폴더 이름에서 숫자 부분을 추출하여 최대값 갱신
                num = int(folder.split("result")[-1])
                max_num = max(max_num, num)
            except ValueError:
                continue
    # 최대 번호 + 1로 새 폴더 이름 생성
    return f"F:/yolo/result{max_num + 1}"

def process_video(video_path, model_path, cam_name, result_folder, device):
    # YOLO 모델을 로드하고 설정합니다
    model = YOLO(model_path)
    model = model.to(device)          # GPU/CPU 설정
    model.conf = 0.2                  # 검출 신뢰도 임계값 설정
    model.imgsz = (960, 960)         # 입력 이미지 크기 설정
    
    # 결과 저장 경로 생성
    save_dir = os.path.join(result_folder, cam_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 입력 비디오 파일 존재 여부 확인
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    # 처리 시작 메시지 출력
    print(f"Processing {cam_name}: {video_path}")
    print(f"Results will be saved to {save_dir}")
    
    # 키포인트 데이터를 저장할 리스트 초기화
    all_keypoints = []
    
    # YOLO 모델로 비디오 처리 시작
    results = model(video_path,
                    task='pose',          # 포즈 인식 작업 설정
                    stream=True,          # 스트리밍 모드 활성화
                    save=True,            # 결과 영상 저장
                    project=save_dir,     # 저장 경로 설정
                    name="",              # 결과 파일 이름
                    device=device,        # 처리 장치 설정
                    verbose=False)        # 상세 출력 비활성화
    
    # 각 프레임별 처리
    for frame_idx, r in enumerate(results):
        # 현재 처리 중인 프레임 번호 출력
        print(f"\r{cam_name} - Processing frame: {frame_idx + 1}", end="")
        # 현재 프레임의 데이터를 저장할 딕셔너리 초기화
        frame_dict = {'frame': frame_idx}
        
        try:
            # 키포인트가 감지된 경우
            if (r.keypoints is not None and 
                len(r.keypoints.data) > 0 and 
                r.keypoints.data[0].shape[0] > 0):
                
                # 첫 번째 감지된 사람의 키포인트 추출
                keypoints = r.keypoints.data[0].cpu().numpy()
                frame_dict['person_detected'] = True
                
                # 각 키포인트의 x, y 좌표와 신뢰도 저장
                for i, name in enumerate(['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                                           'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                                           'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                                           'left_knee', 'right_knee', 'left_ankle', 'right_ankle']):
                    if i < len(keypoints):
                        # 감지된 키포인트 좌표와 신뢰도 저장
                        frame_dict[f'{name}_x'] = float(keypoints[i][0])
                        frame_dict[f'{name}_y'] = float(keypoints[i][1])
                        frame_dict[f'{name}_conf'] = float(keypoints[i][2])
                    else:
                        # 감지되지 않은 키포인트는 0으로 설정
                        frame_dict[f'{name}_x'] = 0.0
                        frame_dict[f'{name}_y'] = 0.0
                        frame_dict[f'{name}_conf'] = 0.0
            else:
                # 사람이 감지되지 않은 경우 모든 값을 0으로 설정
                frame_dict['person_detected'] = False
                for name in ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                             'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                             'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                             'left_knee', 'right_knee', 'left_ankle', 'right_ankle']:
                    frame_dict[f'{name}_x'] = 0.0
                    frame_dict[f'{name}_y'] = 0.0
                    frame_dict[f'{name}_conf'] = 0.0
        
        except Exception as e:
            # 오류 발생 시 에러 메시지 출력하고 모든 값을 0으로 설정
            print(f"\n{cam_name} - Error processing frame {frame_idx}: {str(e)}")
            frame_dict['person_detected'] = False
            for name in ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                         'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                         'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                         'left_knee', 'right_knee', 'left_ankle', 'right_ankle']:
                frame_dict[f'{name}_x'] = 0.0
                frame_dict[f'{name}_y'] = 0.0
                frame_dict[f'{name}_conf'] = 0.0
        
        # 프레임 데이터를 전체 리스트에 추가
        all_keypoints.append(frame_dict)
    
    # 모든 키포인트 데이터를 CSV 파일로 저장
    if all_keypoints:
        df = pd.DataFrame(all_keypoints)
        df.to_csv(f"{save_dir}/keypoints.csv", index=False)
        print(f"\n{cam_name} - Keypoints saved to {save_dir}/keypoints.csv")
    
    # 처리 완료 메시지 출력
    print(f"\n{cam_name} - Total frames processed: {len(all_keypoints)}")



def main():
    # GPU 사용 가능 여부 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print("Using device: cuda")
    else:
        print("CUDA is not available, using CPU")
    
    # 처리할 비디오 파일 경로 설정
    video = r"C:\Users\User\Desktop\Yolo\data\copy_data\video_20250226_114832_408.mp4"
    
    # 결과 저장 폴더 생성
    result_folder = r"C:\Users\User\Desktop\Yolo\copy_result"
    print(f"Using result folder: {result_folder}")
    
    # YOLO 모델 파일 경로 설정
    model_path = r"C:\Users\User\Desktop\Yolo\yolo.pt"

    process_video(video, model_path, "copy_video", result_folder,  device) 
    
  
  
    
    
    # 모든 처리 완료 메시지 출력
    print(f"\nAll videos processed. Results saved to {result_folder}")

# 스크립트가 직접 실행될 때만 main() 함수 실행
if __name__ == "__main__":
    main()
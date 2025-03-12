import os
import torch
from ultralytics import YOLO
import pandas as pd
import numpy as np
import cv2
import time
import threading
import glob

def get_next_result_folder():
    existing_folders = glob.glob("F:/yolo/result*")
    if not existing_folders:
        return "F:/yolo/result"
    max_num = 0
    for folder in existing_folders:
        if folder == "F:/yolo/result":
            max_num = max(max_num, 1)
        else:
            try:
                num = int(folder.split("result")[-1])
                max_num = max(max_num, num)
            except ValueError:
                continue
    return f"F:/yolo/result{max_num + 1}"

def process_video(video_path, model_path, cam_name, result_folder, device):
    model = YOLO(model_path)
    model = model.to(device)
    model.conf = 0.2
    model.imgsz = (960, 960)
    
    save_dir = os.path.join(result_folder, cam_name)
    os.makedirs(save_dir, exist_ok=True)
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    print(f"Processing {cam_name}: {video_path}")
    print(f"Results will be saved to {save_dir}")
    
    all_keypoints = []
    
    results = model(video_path,
                    task='pose',
                    stream=True,
                    save=True,
                    project=save_dir,
                    name="",
                    device=device,
                    verbose=False)
    
    for frame_idx, r in enumerate(results):
        print(f"\r{cam_name} - Processing frame: {frame_idx + 1}", end="")
        frame_dict = {'frame': frame_idx}
        try:
            if (r.keypoints is not None and 
                len(r.keypoints.data) > 0 and 
                r.keypoints.data[0].shape[0] > 0):
                
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
                frame_dict['person_detected'] = False
                for name in ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                             'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                             'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                             'left_knee', 'right_knee', 'left_ankle', 'right_ankle']:
                    frame_dict[f'{name}_x'] = 0.0
                    frame_dict[f'{name}_y'] = 0.0
                    frame_dict[f'{name}_conf'] = 0.0
        except Exception as e:
            print(f"\n{cam_name} - Error processing frame {frame_idx}: {str(e)}")
            frame_dict['person_detected'] = False
            for name in ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                         'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                         'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                         'left_knee', 'right_knee', 'left_ankle', 'right_ankle']:
                frame_dict[f'{name}_x'] = 0.0
                frame_dict[f'{name}_y'] = 0.0
                frame_dict[f'{name}_conf'] = 0.0
        
        all_keypoints.append(frame_dict)
    
    if all_keypoints:
        df = pd.DataFrame(all_keypoints)
        df.to_csv(f"{save_dir}/keypoints.csv", index=False)
        print(f"\n{cam_name} - Keypoints saved to {save_dir}/keypoints.csv")
    
    print(f"\n{cam_name} - Total frames processed: {len(all_keypoints)}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print("Using device: cuda")
    else:
        print("CUDA is not available, using CPU")
    
    video1 = r"F:\yolo\data\cam1\Thread-1_2025-02-20_09-31-18.mp4"
    video2 = r"F:\yolo\data\cam2\Thread-2_2025-02-20_09-31-18.mp4"
    
    result_folder = get_next_result_folder()
    print(f"Using result folder: {result_folder}")
    
    model_path = r"F:/yolo/yolo.pt"
    
    thread1 = threading.Thread(
        target=process_video, 
        args=(video1, model_path, "cam1", result_folder, device)
    )
    
    thread2 = threading.Thread(
        target=process_video, 
        args=(video2, model_path, "cam2", result_folder, device)
    )
    
    thread1.start()
    time.sleep(1)
    thread2.start()
    
    thread1.join()
    thread2.join()
    
    print(f"\nAll videos processed. Results saved to {result_folder}")

if __name__ == "__main__":
    main()

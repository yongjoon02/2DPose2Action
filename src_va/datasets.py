import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# 클래스 매핑 및 레이블 변환 사전
class_mapping = {0: "standing", 1: "sitting", 2: "walking", 3: "no_activity", 4: "no_presence"}
activity_to_label = {"standing": 0, "sitting": 1, "walking": 2, "no_activity": 3, "no_presence": 4}

def _apply_moving_average(coords, window_size=3):
    smoothed = np.copy(coords)
    for i in range(coords.shape[1]):
        smoothed[:, i] = np.convolve(coords[:, i], np.ones(window_size)/window_size, mode='same')
    return smoothed

def _normalize_coords(coords):
    mean = np.mean(coords, axis=0)
    std = np.std(coords, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (coords - mean) / std

class SkeletonDataset(Dataset):
    """
    스켈레톤 데이터셋 클래스 - 프레임 단위로 레이블을 할당합니다.
    """
    def __init__(self, csv_dir, json_dir, transform=None):
        self.csv_dir = csv_dir
        self.json_dir = json_dir
        self.transform = transform
        self.csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
        self.class_mapping = class_mapping
        self.activity_to_label = activity_to_label
        
        # 각 파일의 프레임별 레이블 미리 계산
        self.file_frame_labels = self._compute_frame_labels()
        
        # 클래스별 통계 계산
        self._compute_statistics()
        
    def _compute_frame_labels(self):
        """
        각 CSV 파일의 프레임별 레이블을 계산합니다.
        """
        file_frame_labels = {}
        total_frames = 0
        unknown_activities = set()
        
        print("프레임별 레이블 계산 중...")
        for idx, filename in enumerate(self.csv_files):
            if idx % 20 == 0:
                print(f"처리 중: {idx}/{len(self.csv_files)} 파일")
                
            try:
                # CSV 파일 로드
                csv_path = os.path.join(self.csv_dir, filename)
                df = pd.read_csv(csv_path)
                
                # 'frame' 열이 있는 경우 처리
                if 'frame' in df.columns:
                    df = df[df['frame'] != 'frame']  # 중복된 헤더 행 제거
                    frames = df['frame'].values.astype(np.int64)
                else:
                    frames = np.arange(len(df))
                
                # 최대 프레임 번호
                max_frame = max(frames) if len(frames) > 0 else len(df)
                # 모든 프레임에 기본값 (no_activity) 할당
                labels = np.ones(int(max_frame) + 1, dtype=np.int64) * 3
                
                # JSON 파일에서 활동 정보 로드
                json_path = os.path.join(self.json_dir, filename.replace(".csv", ".json"))
                with open(json_path, 'r') as f:
                    info = json.load(f)
                
                # 리스트 형태인 경우
                if isinstance(info, list):
                    for item in info:
                        if "frameRange" in item and "activity" in item:
                            start_frame, end_frame = item["frameRange"]
                            activity = item["activity"]
                            
                            # 매핑된 레이블 있는지 확인
                            if activity in self.activity_to_label:
                                label = self.activity_to_label[activity]
                                # 해당 프레임 범위에 레이블 할당
                                labels[start_frame:end_frame] = label
                            else:
                                unknown_activities.add(activity)
                
                # 딕셔너리 형태인 경우
                elif isinstance(info, dict) and "activity" in info:
                    activity = info["activity"]
                    if activity in self.activity_to_label:
                        label = self.activity_to_label[activity]
                        # 모든 프레임에 레이블 할당
                        labels[:] = label
                    else:
                        unknown_activities.add(activity)
                
                # 실제 데이터 프레임에 맞는 레이블 추출
                frame_labels = labels[frames]
                file_frame_labels[filename] = (frames, frame_labels)
                total_frames += len(frames)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                # 오류 시 모든 프레임에 no_activity 할당
                try:
                    csv_path = os.path.join(self.csv_dir, filename)
                    df = pd.read_csv(csv_path)
                    frames = np.arange(len(df))
                    frame_labels = np.ones(len(frames), dtype=np.int64) * 3
                    file_frame_labels[filename] = (frames, frame_labels)
                except:
                    pass
        
        # 처리 결과 통계
        if unknown_activities:
            print(f"알 수 없는 활동 유형: {unknown_activities}")
        print(f"총 {len(file_frame_labels)}개 파일, {total_frames}개 프레임 처리됨")
        
        return file_frame_labels
    
    def _compute_statistics(self):
        """
        데이터셋 통계 계산 (클래스별 프레임 수 등)
        """
        self.class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        total_frames = 0
        
        for filename, (frames, labels) in self.file_frame_labels.items():
            for label in labels:
                self.class_counts[label] += 1
                total_frames += 1
        
        # 클래스 분포 출력
        print("\n프레임 단위 클래스 분포:")
        for cls_idx, count in self.class_counts.items():
            percent = (count / total_frames) * 100 if total_frames > 0 else 0
            print(f"  {cls_idx} ({class_mapping[cls_idx]}): {count} 프레임 ({percent:.2f}%)")
    
    def __len__(self):
        return len(self.csv_files)
    
    def __getitem__(self, idx):
        try:
            filename = self.csv_files[idx]
            csv_path = os.path.join(self.csv_dir, filename)
            df = pd.read_csv(csv_path)
            
            # 'frame' 열이 있는 경우 처리
            if 'frame' in df.columns:
                df = df[df['frame'] != 'frame']  # 중복된 헤더 행 제거
                joint_columns = [col for col in df.columns if col != 'frame']
            else:
                joint_columns = df.columns
            
            # 좌표 데이터 추출 및 전처리
            coords = df[joint_columns].values.astype(np.float32)
            coords = _apply_moving_average(coords, window_size=3)
            coords = _normalize_coords(coords)
            
            # 프레임별 레이블 가져오기
            _, frame_labels = self.file_frame_labels[filename]
            
            # 텐서로 변환
            coords = torch.tensor(coords, dtype=torch.float32)
            frame_labels = torch.tensor(frame_labels, dtype=torch.long)
            
            # 변환 적용
            if self.transform:
                coords = self.transform(coords)
            
            return coords, frame_labels
        
        except Exception as e:
            print(f"Error loading {self.csv_files[idx]}: {e}")
            # 오류 시 더미 데이터 반환
            coords = torch.zeros((100, 34), dtype=torch.float32)
            labels = torch.ones(100, dtype=torch.long) * 3  # no_activity
            return coords, labels
    
    def get_class_distribution(self):
        """
        클래스별 프레임 수 반환
        """
        return self.class_counts
    
    def get_file_class_counts(self):
        """
        각 파일별 클래스 분포 반환
        """
        file_class_counts = {}
        for filename, (frames, labels) in self.file_frame_labels.items():
            counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            for label in labels:
                counts[label] += 1
            file_class_counts[filename] = counts
        return file_class_counts

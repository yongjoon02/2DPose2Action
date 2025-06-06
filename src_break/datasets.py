import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# Breakfast 데이터셋의 coarse 액션 클래스 매핑
breakfast_class_mapping = {
    'sil': 0,      # 정지/background
    'cut': 1,      # 자르기
    'put': 2,      # 놓기
    'crack': 3,    # 깨기 (달걀 등)
    'stir': 4,     # 젓기
    'add': 5,      # 추가하기
    'butter': 6,   # 버터 바르기
    'pour': 7,     # 붓기
    'fry': 8,      # 튀기기/볶기
    'take': 9,     # 가져오기
    'spoon': 10,   # 숟가락질
}

# 역방향 매핑
class_to_name = {v: k for k, v in breakfast_class_mapping.items()}

def _apply_moving_average(coords, window_size=3):
    """포즈 좌표에 이동 평균 적용하여 스무딩"""
    smoothed = np.copy(coords)
    for i in range(coords.shape[1]):
        smoothed[:, i] = np.convolve(coords[:, i], np.ones(window_size)/window_size, mode='same')
    return smoothed

def _normalize_coords(coords):
    """포즈 좌표 정규화"""
    mean = np.mean(coords, axis=0)
    std = np.std(coords, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (coords - mean) / std

def _parse_label_file(label_path):
    """라벨 파일 파싱: start-end action 형식"""
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                time_range, action = parts[0], parts[1]
                if '-' in time_range:
                    start, end = map(int, time_range.split('-'))
                    labels.append((start, end, action))
    return labels

class BreakfastSkeletonDataset(Dataset):
    """Breakfast 데이터셋을 위한 포즈 기반 액션 인식 데이터셋"""
    
    def __init__(self, skeleton_dir, label_dir, transform=None, training=True, use_augmentation=True, sequence_length=60):
        self.skeleton_dir = skeleton_dir  # 포즈 CSV 파일들이 있는 디렉토리
        self.label_dir = label_dir        # 라벨 파일들이 있는 디렉토리
        self.transform = transform
        self.training = training
        self.use_augmentation = use_augmentation and training
        self.sequence_length = sequence_length  # 시퀀스 길이
        
        # 사용 가능한 파일들 찾기
        self.data_pairs = self._find_matching_files()
        
        print(f"총 {len(self.data_pairs)}개의 데이터 파일 쌍을 발견했습니다.")
        if len(self.data_pairs) > 0:
            print(f"예시: {self.data_pairs[0]}")
    
    def _find_matching_files(self):
        """포즈 CSV 파일과 라벨 파일 매칭"""
        pairs = []
        
        # 포즈 CSV 파일들 찾기
        if not os.path.exists(self.skeleton_dir):
            print(f"경고: 포즈 디렉토리가 존재하지 않습니다: {self.skeleton_dir}")
            return pairs
            
        csv_files = [f for f in os.listdir(self.skeleton_dir) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            # CSV 파일명에서 라벨 파일명 생성 (예: P52_salat.csv -> P52_salat.labels)
            base_name = csv_file.replace('.csv', '')
            label_file = base_name + '.labels'
            
            csv_path = os.path.join(self.skeleton_dir, csv_file)
            label_path = os.path.join(self.label_dir, label_file)
            
            if os.path.exists(label_path):
                pairs.append((csv_path, label_path, base_name))
            else:
                print(f"경고: {label_file}에 대응하는 라벨 파일을 찾을 수 없습니다.")
        
        return pairs
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        csv_path, label_path, video_name = self.data_pairs[idx]
        
        # 포즈 데이터 로드
        df = pd.read_csv(csv_path)
        
        # 프레임 컬럼 처리
        if 'frame' in df.columns:
            frames = df['frame'].values.astype(np.int64)
            joint_columns = [col for col in df.columns if col != 'frame' and col != 'person_detected']
        else:
            frames = np.arange(len(df))
            joint_columns = [col for col in df.columns if col != 'person_detected']
        
        # 관절 좌표 추출
        coords = df[joint_columns].values.astype(np.float32)
        
        # 포즈 데이터 전처리
        coords = _apply_moving_average(coords, window_size=3)
        coords = _normalize_coords(coords)
        
        # 라벨 데이터 로드
        label_segments = _parse_label_file(label_path)
        
        # 프레임별 라벨 생성
        max_frame = len(coords)
        frame_labels = np.zeros(max_frame, dtype=np.int64)  # 기본값: sil (0)
        
        for start, end, action in label_segments:
            # 프레임 인덱스에 맞게 조정 (1-based -> 0-based)
            start_idx = max(0, start - 1)
            end_idx = min(max_frame, end)
            
            if action in breakfast_class_mapping:
                label_idx = breakfast_class_mapping[action]
                frame_labels[start_idx:end_idx] = label_idx
        
        # 데이터 증강 적용 (훈련 시)
        if self.use_augmentation:
            from src_break.augmentations import augment_skeleton_data
            coords = augment_skeleton_data(coords, frames)
        
        # 시퀀스 길이에 맞게 조정
        if len(coords) > self.sequence_length:
            # 긴 시퀀스는 랜덤하게 자르기
            start_idx = np.random.randint(0, len(coords) - self.sequence_length + 1) if self.training else 0
            coords = coords[start_idx:start_idx + self.sequence_length]
            frame_labels = frame_labels[start_idx:start_idx + self.sequence_length]
        elif len(coords) < self.sequence_length:
            # 짧은 시퀀스는 패딩
            pad_length = self.sequence_length - len(coords)
            coords = np.pad(coords, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
            frame_labels = np.pad(frame_labels, (0, pad_length), mode='constant', constant_values=0)
        
        coords = torch.tensor(coords, dtype=torch.float32)
        frame_labels = torch.tensor(frame_labels, dtype=torch.long)
        
        return coords, frame_labels, video_name
    
    def get_class_distribution(self):
        """클래스 분포 계산"""
        class_counts = {i: 0 for i in range(len(breakfast_class_mapping))}
        
        for csv_path, label_path, _ in self.data_pairs:
            label_segments = _parse_label_file(label_path)
            for start, end, action in label_segments:
                if action in breakfast_class_mapping:
                    class_counts[breakfast_class_mapping[action]] += (end - start)
        
        return class_counts
    
    def get_class_names(self):
        """클래스 이름 반환"""
        return [class_to_name[i] for i in range(len(breakfast_class_mapping))]
    
    @property
    def class_mapping(self):
        """클래스 매핑 반환"""
        return breakfast_class_mapping


# 기존 코드와의 호환성을 위한 alias
SkeletonDataset = BreakfastSkeletonDataset
class_mapping = breakfast_class_mapping

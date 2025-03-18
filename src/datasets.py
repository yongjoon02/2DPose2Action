import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# 클래스 매핑 및 레이블 변환 사전
class_mapping = {0: "standing", 1: "sitting", 2: "walking", 3: "no_activity", 4: "no_presence"}
activity_to_label = {"standing": 0, "sitting": 1, "walking": 2, "no_activity": 3, "no_presence": 4, "no presence": 4}

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
    def __init__(self, csv_dir, json_dir, transform=None, training=True, use_augmentation=True):
        self.csv_dir = csv_dir
        self.json_dir = json_dir
        self.transform = transform
        self.training = training
        self.use_augmentation = use_augmentation and training
        self.csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
        self.file_labels = self._compute_file_labels()
    
    def _compute_file_labels(self):
        file_labels = {}
        for idx in range(len(self.csv_files)):
            csv_path = os.path.join(self.csv_dir, self.csv_files[idx])
            df = pd.read_csv(csv_path)
            json_path = os.path.join(self.json_dir, self.csv_files[idx].replace(".csv", ".json"))
            with open(json_path, 'r') as f:
                info = json.load(f)
            main_label = 3  # 기본값
            max_duration = 0
            if isinstance(info, list):
                for item in info:
                    if "frameRange" in item and "activity" in item:
                        start_frame, end_frame = item["frameRange"]
                        duration = end_frame - start_frame
                        if duration > max_duration and item["activity"] in activity_to_label:
                            max_duration = duration
                            main_label = activity_to_label[item["activity"]]
            file_labels[self.csv_files[idx]] = main_label
        return file_labels
    
    def __len__(self):
        return len(self.csv_files)
    
    def __getitem__(self, idx):
        csv_path = os.path.join(self.csv_dir, self.csv_files[idx])
        df = pd.read_csv(csv_path)
        if 'frame' in df.columns:
            df = df[df['frame'] != 'frame']
            frames = df['frame'].values.astype(np.int64)
            joint_columns = [col for col in df.columns if col != 'frame']
        else:
            frames = np.arange(len(df))
            joint_columns = df.columns
        
        coords = df[joint_columns].values.astype(np.float32)
        coords = _apply_moving_average(coords, window_size=3)
        coords = _normalize_coords(coords)
        
        # 증강은 augmentations 모듈에서 가져오기
        if self.use_augmentation:
            from src.augmentations import augment_skeleton_data
            coords = augment_skeleton_data(coords, frames)
        
        coords = torch.tensor(coords, dtype=torch.float32)
        
        json_path = os.path.join(self.json_dir, self.csv_files[idx].replace(".csv", ".json"))
        with open(json_path, 'r') as f:
            info = json.load(f)
        
        max_frame = max(frames) if len(frames) > 0 else len(df)
        labels = np.ones(int(max_frame) + 1, dtype=np.int64) * 3
        if isinstance(info, list):
            for item in info:
                if "frameRange" in item and "activity" in item:
                    start_frame, end_frame = item["frameRange"]
                    label = activity_to_label.get(item["activity"], 3)
                    labels[start_frame:end_frame] = label
        
        frame_labels = labels[frames]
        frame_labels = torch.tensor(frame_labels, dtype=torch.long)
        return coords, frame_labels, self.csv_files[idx]
    
    def get_class_distribution(self):
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for label in self.file_labels.values():
            class_counts[label] += 1
        return class_counts

    def get_labels(self):
        return [self.file_labels[f] for f in self.csv_files]

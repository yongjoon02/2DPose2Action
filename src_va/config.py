import os
import torch

def get_config():
    # 프로젝트 루트 디렉토리 가져오기
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir)
    
    config = {
        "model": {
            "input_size": 34,  # 스켈레톤 관절 좌표 개수
            "num_channels": [64, 64, 64, 64, 64],  # 각 TCN 블록의 채널 수
            "kernel_size": 3,  # 컨볼루션 커널 크기
            "dropout": 0.2,    # 드롭아웃 비율
            "num_classes": 5   # 클래스 개수 (standing, sitting, walking, no_activity, no_presence)
        },
        "training": {
            "batch_size": 32,
            "num_epochs": 200,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        "data": {
            "csv_dir": os.path.join(project_root, "data"),
            "json_dir": os.path.join(project_root, "data")
        },
        "class_labels": ["standing", "sitting", "walking", "no_activity", "no_presence"]
    }
    
    return config 
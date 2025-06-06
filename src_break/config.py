import os
import sys
import time
import torch
import numpy as np

def get_training_config(dataset, input_size, num_classes, hidden_channels, kernel_size, 
                       dropout, class_mapping, device, module_files, src_dir):
    return {
        "training": {
            "batch_size": 16,  # Breakfast 데이터셋은 시퀀스가 길어서 배치 크기 줄임
            "num_epochs": 150,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "early_stopping_patience": 25,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingWarmRestarts",
            "scheduler_params": {
                "T_0": 10,
                "T_mult": 2
            },
            "loss_function": "WeightedFocalLoss",
            "loss_params": {
                "gamma": 2.0,
                "class_weights": None,  # 실제 학습 시 클래스 분포에 따라 설정
                "weight_description": "Breakfast 액션 클래스 불균형 해소를 위한 가중치"
            },
            "augmentation": {
                "enabled": dataset.use_augmentation,
                "augment_probability": 0.6,  # Breakfast 데이터에 맞게 조정
                "techniques": [
                    "time_warping", "noise_addition", "rotation", "scaling"
                ]
            }
        },
        "model": {
            "type": "TCN",
            "input_size": input_size,
            "output_size": num_classes,
            "hidden_channels": hidden_channels,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "use_se": True,
            "sequence_length": 60,  # Breakfast 데이터셋의 시퀀스 길이
            "transition_rules": {
                "enabled": True,
                "description": "Breakfast 액션 전이 규칙",
                "forbidden_transitions": [
                    # 예: sil에서 바로 복잡한 액션으로 가는 것을 제한
                    {"from": "sil", "to": "fry", "from_idx": 0, "to_idx": 8},
                ],
                "method": "FSM with probability-based selection"
            }
        },
        "classes": ["sil", "cut", "put", "crack", "stir", "add", "butter", "pour", "fry", "take", "spoon"],  # Breakfast 액션 클래스
        "device": str(device),
        "seed": 42,
        "environment": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "numpy_version": np.__version__,
            "code_archived": True,
            "archived_files": [
                "train_break.py", 
                *[f for f in module_files if os.path.exists(os.path.join(src_dir, f))]
            ]
        }
    } 
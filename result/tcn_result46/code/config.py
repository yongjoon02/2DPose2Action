import os
import sys
import time
import torch
import numpy as np

def get_training_config(dataset, input_size, num_classes, hidden_channels, kernel_size, 
                       dropout, class_mapping, device, module_files, src_dir):
    return {
        "training": {
            "batch_size": 32,
            "num_epochs": 10,
            "learning_rate": 0.001,
            "weight_decay": 0.001,
            "num_folds": 3,
            "early_stopping_patience": 20,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingWarmRestarts",
            "scheduler_params": {
                "T_0": 10,
                "T_mult": 2
            },
            "loss_function": "WeightedFocalLoss",
            "loss_params": {
                "gamma": 2.0,
                "class_weights": None,  # 실제 학습 시 설정
                "weight_description": "클래스 불균형 해소 및 중요 클래스(standing, sitting) 강조를 위한 가중치"
            },
            "augmentation": {
                "enabled": dataset.use_augmentation,
                "augment_probability": 0.7,
                "techniques": [
                    # ... (기존 augmentation 설정)
                ],
                "enhanced_augmentation": {
                    # ... (기존 enhanced_augmentation 설정)
                }
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
            "transition_rules": {
                "enabled": True,
                "description": "FSM과 확률 기반 접근법을 결합한 전이 규칙",
                "forbidden_transitions": [
                    {"from": "sitting", "to": "walking", "from_idx": 1, "to_idx": 2},
                    {"from": "walking", "to": "standing", "from_idx": 2, "to_idx": 0}
                ],
                "method": "FSM with probability-based selection"
            }
        },
        "classes": list(class_mapping.values()),
        "device": str(device),
        "seed": 42,
        "environment": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "numpy_version": np.__version__,
            "code_archived": True,
            "archived_files": [
                "train.py", 
                *[f for f in module_files if os.path.exists(os.path.join(src_dir, f))]
            ]
        }
    } 
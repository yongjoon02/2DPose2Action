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
            "num_epochs": 200,
            "learning_rate": 0.001,
            "weight_decay": 0.001,
            "early_stopping_patience": 20
        },
        "model": {
            "type": "TCN",
            "input_size": input_size,
            "output_size": num_classes,
            "hidden_channels": hidden_channels,
            "kernel_size": kernel_size,
            "dropout": dropout
        },
        "classes": list(class_mapping.values()),
        "device": str(device),
        "seed": 42
    } 
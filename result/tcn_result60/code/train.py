import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
# 상위 디렉토리를 path에 추가하여 src 패키지를 인식하도록 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.datasets import SkeletonDataset, class_mapping
from src.trainer import train_model  # 학습 관련 기능은 모두 trainer로 이동
from src.config import get_training_config  # 설정 관련 기능
from src.utils import set_seed  # 유틸리티 함수

def main():
    # 기본 설정
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터 경로
    csv_dir = "data/csv"
    json_dir = "data/json"
    
    input_size = 34
    num_classes = 5
    hidden_channels = [64, 128, 256, 512, 256, 128]
    kernel_size = 7
    dropout = 0.25
    
    # 데이터셋 생성
    dataset = SkeletonDataset(csv_dir=csv_dir, json_dir=json_dir, use_augmentation=True)
    
    # 설정 가져오기
    config = get_training_config(
        dataset=dataset,
        input_size=input_size,
        num_classes=num_classes,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        class_mapping=class_mapping,
        device=device,
        module_files=["models.py", "datasets.py", "losses.py", "augmentations.py"],
        src_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    )
    
    # 모델 학습 및 결과 저장 - trainer 모듈의 train_model 함수 사용
    result_dir, results = train_model(dataset, config)
    print(f"학습 완료! 결과는 {result_dir} 디렉토리에 저장되었습니다.")
    return result_dir, results

if __name__ == "__main__":
    main()

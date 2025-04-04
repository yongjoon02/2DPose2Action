import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import click
# 상위 디렉토리를 path에 추가하여 src 패키지를 인식하도록 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.datasets import SkeletonDataset, class_mapping
from src.trainer import train_model
from src.config import get_training_config
from src.utils import set_seed

@click.command()
@click.option('--dataset-size', type=click.Choice(['500', '1000', '2000']), required=True, help='사용할 데이터셋 크기 선택')
@click.option('--use-cpu', is_flag=True, help='CPU 사용 여부 (기본값: False)')
def main(dataset_size, use_cpu):
    """모델 학습 실행"""
    # 시드 초기화
    set_seed()
    
    # 데이터셋 크기에 따른 경로 설정
    csv_dir = f"data/csv_{dataset_size}"
    json_dir = f"data/json_{dataset_size}"
    
    # 디렉토리 존재 확인
    if not os.path.exists(csv_dir) or not os.path.exists(json_dir):
        print(f"오류: {csv_dir} 또는 {json_dir} 디렉토리를 찾을 수 없습니다.")
        return
    
    print(f"선택된 데이터셋 크기: {dataset_size}")
    print(f"CSV 디렉토리: {csv_dir}")
    print(f"JSON 디렉토리: {json_dir}")
    
    # 디바이스 설정
    if use_cpu:
        device = torch.device("cpu")
        print("CPU를 사용하여 학습을 진행합니다.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{device}를 사용하여 학습을 진행합니다.")
    
    # 데이터셋 생성
    dataset = SkeletonDataset(
        csv_dir=csv_dir,
        json_dir=json_dir,
        transform=None,
        training=True,
        use_augmentation=True
    )
    
    # 설정 가져오기
    config = get_training_config(
        dataset=dataset,
        input_size=dataset[0][0].shape[1],  # feature_dim
        num_classes=len(dataset.class_mapping),
        hidden_channels=[64, 128, 256],
        kernel_size=3,
        dropout=0.2,
        class_mapping=dataset.class_mapping,
        device=device,
        module_files=["src/models.py", "src/losses.py", "src/datasets.py"],
        src_dir="src"
    )
    
    # 모델 학습
    result_dir, final_results = train_model(dataset, config)
    
    print("\n최종 결과:")
    print(f"검증 정확도: {final_results['val_accuracy']:.2f}%")
    print(f"검증 손실: {final_results['val_loss']:.6f}")
    print(f"결과가 {result_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main()

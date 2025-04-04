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
@click.option('--dataset-size', type=click.Choice(['500', '1000', '2000']), required=True, help='데이터셋 크기 선택 (500/1000/2000)')
def main(dataset_size):
    # 시드 설정
    set_seed()
    
    # 입력 크기와 클래스 수 설정
    input_size = 34  # 17개 관절 x 2 (x, y 좌표)
    num_classes = 5  # 클래스 수
    
    # 데이터셋 경로 설정
    csv_dir = f"data/csv_{dataset_size}"
    json_dir = f"data/json_{dataset_size}"
    
    # 디렉토리 존재 여부 확인
    if not os.path.exists(csv_dir) or not os.path.exists(json_dir):
        print(f"오류: {dataset_size} 크기의 데이터셋 디렉토리를 찾을 수 없습니다.")
        print(f"확인할 경로: {csv_dir}, {json_dir}")
        return
    
    print(f"선택된 데이터셋 크기: {dataset_size}")
    print(f"CSV 디렉토리: {csv_dir}")
    print(f"JSON 디렉토리: {json_dir}")
    
    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device}를 사용합니다.")
    
    # 데이터셋 생성
    dataset = SkeletonDataset(
        csv_dir=csv_dir,
        json_dir=json_dir,
        transform=None,
        training=True,
        use_augmentation=True
    )
    
    print(f"데이터셋 크기: {len(dataset)}")
    
    # 학습 설정 가져오기
    config = get_training_config(
        dataset=dataset,
        input_size=input_size,
        num_classes=num_classes,
        hidden_channels=[64, 128, 256, 512, 256, 128],
        kernel_size=7,
        dropout=0.25,
        class_mapping=class_mapping,
        device=device,
        module_files=["models.py", "datasets.py", "losses.py", "augmentations.py"],
        src_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    )
    
    # 모델 학습
    result_dir, final_results = train_model(dataset, config)
    
    # 결과 출력
    print("\n최종 결과:")
    print(f"검증 정확도: {final_results['val_accuracy']:.2f}%")
    print(f"검증 손실: {final_results['val_loss']:.6f}")
    print(f"결과가 {result_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main()

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import click
# 상위 디렉토리를 path에 추가하여 src_break 패키지를 인식하도록 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src_break.datasets import BreakfastSkeletonDataset
from src_break.trainer import train_model
from src_break.config import get_training_config
from src_break.utils import set_seed

@click.command()
@click.option('--skeleton-dir', default='result/breakfast_skeleton_result_prepro', help='포즈 CSV 파일이 있는 디렉토리')
@click.option('--use-cpu', is_flag=True, help='CPU 사용 여부 (기본값: False)')
@click.option('--sequence-length', default=60, help='입력 시퀀스 길이')
def main(skeleton_dir, use_cpu, sequence_length):
    """Breakfast 데이터셋으로 액션 인식 모델 학습"""
    # 시드 초기화
    set_seed()
    
    # 경로 설정
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    skeleton_full_path = os.path.join(project_root, skeleton_dir)
    label_dir = os.path.join(project_root, "data", "Breakfast", "labels", "coarse")
    
    print(f"포즈 데이터 디렉토리: {skeleton_full_path}")
    print(f"라벨 디렉토리: {label_dir}")
    
    # 디렉토리 존재 확인
    if not os.path.exists(skeleton_full_path):
        print(f"오류: 포즈 데이터 디렉토리를 찾을 수 없습니다: {skeleton_full_path}")
        print("먼저 yolo_pose_csv_kdy.py를 실행하여 포즈 데이터를 생성해주세요.")
        return
        
    if not os.path.exists(label_dir):
        print(f"오류: 라벨 디렉토리를 찾을 수 없습니다: {label_dir}")
        return
    
    # 디바이스 설정
    if use_cpu:
        device = torch.device("cpu")
        print("CPU를 사용하여 학습을 진행합니다.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{device}를 사용하여 학습을 진행합니다.")
    
    # 데이터셋 생성
    print("데이터셋을 로드하는 중...")
    dataset = BreakfastSkeletonDataset(
        skeleton_dir=skeleton_full_path,
        label_dir=label_dir,
        transform=None,
        training=True,
        use_augmentation=True,
        sequence_length=sequence_length
    )
    
    if len(dataset) == 0:
        print("오류: 유효한 데이터 파일이 없습니다.")
        return
    
    print(f"총 {len(dataset)}개의 데이터 샘플이 로드되었습니다.")
    
    # 클래스 분포 확인
    class_distribution = dataset.get_class_distribution()
    class_names = dataset.get_class_names()
    
    print("\n클래스 분포:")
    for i, (class_idx, count) in enumerate(class_distribution.items()):
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
        print(f"  {class_name}: {count}개 프레임")
    
    # 첫 번째 샘플로 입력 크기 확인
    sample_coords, sample_labels, sample_name = dataset[0]
    feature_dim = sample_coords.shape[1]  # (sequence_length, feature_dim)
    num_classes = len(dataset.class_mapping)
    
    print(f"\n데이터 정보:")
    print(f"  입력 특성 차원: {feature_dim}")
    print(f"  시퀀스 길이: {sequence_length}")
    print(f"  클래스 수: {num_classes}")
    print(f"  샘플 예시: {sample_name}")
    
    # 설정 가져오기
    config = get_training_config(
        dataset=dataset,
        input_size=feature_dim,
        num_classes=num_classes,
        hidden_channels=[64, 128, 256, 512],  # Breakfast 데이터에 맞게 더 큰 모델
        kernel_size=3,
        dropout=0.3,
        class_mapping=dataset.class_mapping,
        device=device,
        module_files=["models.py", "losses.py", "datasets.py", "augmentations.py"],
        src_dir="src_break"
    )
    
    # 데이터셋 정보 추가
    config["dataset"] = {
        "name": "Breakfast",
        "skeleton_dir": skeleton_dir,
        "label_dir": "data/Breakfast/labels/coarse",
        "num_samples": len(dataset),
        "sequence_length": sequence_length,
        "feature_dim": feature_dim,
        "class_distribution": class_distribution,
        "class_names": class_names,
        "use_augmentation": dataset.use_augmentation
    }
    
    print("\n학습을 시작합니다...")
    
    # 모델 학습
    result_dir, final_results = train_model(dataset, config)
    
    print("\n=== 최종 결과 ===")
    print(f"검증 정확도: {final_results['val_accuracy']:.2f}%")
    print(f"검증 손실: {final_results['val_loss']:.6f}")
    print(f"결과가 {result_dir}에 저장되었습니다.")
    
    # 클래스별 성능 정보 출력 (있는 경우)
    if 'class_metrics' in final_results:
        print("\n클래스별 성능:")
        for i, metrics in final_results['class_metrics'].items():
            class_name = class_names[i] if i < len(class_names) else f"class_{i}"
            print(f"  {class_name}: 정밀도={metrics['precision']:.3f}, 재현율={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

if __name__ == "__main__":
    main()

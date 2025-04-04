import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import argparse

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src_va.config import get_config
from src_va.models import TCN
from src_va.datasets import SkeletonDataset, class_mapping
from src_va.trainer import train_model, validate, custom_collate, evaluate_and_save_test_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train TCN model for video activity recognition')
    parser.add_argument('--dataset-size', type=int, choices=[500, 1000, 2000], 
                        help='Size of the dataset to use (500, 1000, or 2000)')
    parser.add_argument('--use-cpu', action='store_true', 
                        help='Force using CPU instead of GPU')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate on test dataset using the latest saved model')
    args = parser.parse_args()

    # Get configuration
    config = get_config()
    
    # 설정 업데이트
    config["training"]["learning_rate"] = args.lr
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    if args.use_cpu:
        config["training"]["device"] = "cpu"
    
    # Set data directories based on dataset size
    if args.dataset_size:
        config["data"]["csv_dir"] = os.path.join(project_root, "data", f"csv_{args.dataset_size}")
        config["data"]["json_dir"] = os.path.join(project_root, "data", f"json_{args.dataset_size}")
    else:
        print("No dataset size specified. Using default size: 500")
        config["data"]["csv_dir"] = os.path.join(project_root, "data", "csv_500")
        config["data"]["json_dir"] = os.path.join(project_root, "data", "json_500")
    
    print(f"Using CSV directory: {config['data']['csv_dir']}")
    print(f"Using JSON directory: {config['data']['json_dir']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    
    # 테스트 데이터셋 로드
    test_dir = os.path.join(project_root, "data", "test_data")
    try:
        test_dataset = SkeletonDataset(
            csv_dir=os.path.join(test_dir, "test_csv"),
            json_dir=os.path.join(test_dir, "test_json"),
            transform=None
        )
        print(f"Loaded {len(test_dataset)} test sequences")
        
        # 테스트 세트 클래스 분포 확인
        test_class_dist = test_dataset.get_class_distribution()
        print("\nTest set class distribution:")
        for class_idx, count in test_class_dist.items():
            if count > 0:
                print(f"  {class_mapping[class_idx]}: {count} samples")
                
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            collate_fn=custom_collate,
            num_workers=2
        )
    except Exception as e:
        print(f"Error creating test dataset: {e}")
        print(f"Cannot proceed without test dataset")
        sys.exit(1)
    
    # 평가 모드만 실행할 경우
    if args.eval_only:
        print("Evaluation only mode")
        # 최근 모델 찾기
        result_dirs = [d for d in os.listdir("result") if os.path.isdir(os.path.join("result", d)) and d.startswith("va_tcn_result")]
        if not result_dirs:
            print("No trained model found. Please train a model first.")
            sys.exit(1)
            
        latest_result_dir = os.path.join("result", sorted(result_dirs)[-1])
        model_path = os.path.join(latest_result_dir, "best_model.pth")
        
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            sys.exit(1)
            
        # 모델 생성 및 로드
        model = TCN(
            input_size=config["model"]["input_size"],
            num_channels=config["model"]["num_channels"],
            kernel_size=config["model"]["kernel_size"],
            dropout=config["model"]["dropout"],
            num_classes=config["model"]["num_classes"]
        ).to(config["training"]["device"])
        
        model.load_state_dict(torch.load(model_path, map_location=config["training"]["device"]))
        print(f"Loaded model from {model_path}")
        
        # 테스트 데이터셋에 대한 평가 및 결과 저장
        criterion = nn.CrossEntropyLoss()
        test_results = evaluate_and_save_test_results(model, test_loader, criterion, config, latest_result_dir)
        print(f"Test accuracy: {test_results['accuracy']:.4f}")
        print(f"Results saved to {latest_result_dir}")
        sys.exit(0)
    
    # 학습 데이터셋 로드
    try:
        dataset = SkeletonDataset(
            csv_dir=config["data"]["csv_dir"],
            json_dir=config["data"]["json_dir"],
            transform=None
        )
        print(f"Loaded {len(dataset)} sequence files")
        
        # Print class distribution
        class_dist = dataset.get_class_distribution()
        print("Class distribution:")
        for class_idx, count in class_dist.items():
            print(f"  {class_mapping[class_idx]}: {count} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # 학습/검증 세트 분할
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    
    train_size = int(0.8 * len(dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 데이터 로더 생성
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=2
    )
    
    # Create model
    model = TCN(
        input_size=config["model"]["input_size"],
        num_channels=config["model"]["num_channels"],
        kernel_size=config["model"]["kernel_size"],
        dropout=config["model"]["dropout"],
        num_classes=config["model"]["num_classes"]
    ).to(config["training"]["device"])
    
    # 모델 가중치 초기화
    def init_weights(m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
    
    model.apply(init_weights)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=1e-5  # 가중치 감쇠 작게 설정
    )
    
    # 학습률 스케줄러 추가
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Print model summary
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Training on device: {config['training']['device']}")
    print(f"Class mapping: {class_mapping}")
    
    # Train model
    result_dir = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        save_validation_results=False  # 검증 결과는 저장하지 않음
    )
    
    # 테스트 데이터셋에 대한 평가 및 결과 저장
    print("\nEvaluating on test set...")
    test_results = evaluate_and_save_test_results(model, test_loader, criterion, config, result_dir)
    print(f"Test accuracy: {test_results['accuracy']:.4f}")
    
    print("Training and evaluation completed!")

if __name__ == "__main__":
    main() 
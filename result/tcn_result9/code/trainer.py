import os
import torch
import numpy as np
import json
import csv
import time
import shutil
import glob
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import confusion_matrix, classification_report

from .models import TCN
from .losses import WeightedFocalLoss
from .datasets import class_mapping

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(dataloader, desc="훈련 중"):
        coords, labels, _ = batch
        coords = coords.to(device, dtype=torch.float32)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(coords, conservative_no_activity=False, apply_transition_rules=False)
        
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        B, T, _ = outputs.shape
        outputs = outputs.reshape(B * T, -1)
        labels = labels.reshape(B * T)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="검증 중"):
            coords, labels, _ = batch
            coords = coords.to(device, dtype=torch.float32)
            labels = labels.to(device)
            
            outputs = model(coords, conservative_no_activity=False, apply_transition_rules=False)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            B, T, _ = outputs.shape
            outputs = outputs.reshape(B * T, -1)
            labels = labels.reshape(B * T)
            
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return epoch_loss / len(dataloader), 100 * correct / total

def train_with_early_stopping(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, 
                           num_epochs=200, patience=20, fold_dir=None):
    best_valid_acc = 0
    best_model_path = os.path.join(fold_dir, "best_model.pth")
    counter = 0
    
    train_loss_history = []
    valid_loss_history = []
    valid_acc_history = []
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc = validate(model, valid_loader, criterion, device)
        
        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        valid_acc_history.append(valid_acc)
        
        if scheduler is not None:
            scheduler.step()
        
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%, 시간: {elapsed:.2f}초")
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), best_model_path)
            counter = 0
        else:
            counter += 1
        
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}, 최고 검증 정확도: {best_valid_acc:.2f}%")
            break
    
    if fold_dir:
        save_learning_curves(
            train_loss_history,
            valid_loss_history,
            valid_acc_history,
            os.path.join(fold_dir, "learning_curves.csv")
        )
    
    model.load_state_dict(torch.load(best_model_path))
    return model, best_valid_acc

def create_result_dir():
    """결과 저장을 위한 고유한 디렉토리 이름 생성"""
    base_dir = "result"
    base_name = "tcn_result"
    
    # 기본 디렉토리가 없으면 생성
    os.makedirs(base_dir, exist_ok=True)
    
    # 기존 tcn_result 디렉토리 확인
    existing_dirs = glob.glob(os.path.join(base_dir, f"{base_name}*"))
    
    if not existing_dirs:
        result_dir = os.path.join(base_dir, base_name)
    else:
        # 숫자 접미사가 있는 디렉토리들 찾기
        indices = []
        for dir_path in existing_dirs:
            dir_name = os.path.basename(dir_path)
            if dir_name == base_name:
                indices.append(0)
            elif dir_name.startswith(base_name) and dir_name[len(base_name):].isdigit():
                indices.append(int(dir_name[len(base_name):]))
        
        # 다음 인덱스 결정
        next_index = max(indices) + 1 if indices else 1
        result_dir = os.path.join(base_dir, f"{base_name}{next_index}")
    
    # 결과 디렉토리 생성
    os.makedirs(result_dir, exist_ok=True)
    print(f"결과 저장 디렉토리: {result_dir}")
    
    return result_dir

def save_source_code(dest_dir):
    """현재 사용 중인 코드 파일들을 저장"""
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 코드 디렉토리 생성 (src_files 대신 code 폴더 사용)
    code_dir = os.path.join(dest_dir, "code")
    os.makedirs(code_dir, exist_ok=True)
    
    # src 디렉토리 내의 Python 파일들
    for file in glob.glob(os.path.join(src_dir, "src", "*.py")):
        shutil.copy2(file, code_dir)
    
    # train.py 파일 복사
    train_py = os.path.join(src_dir, "script", "train.py")
    if os.path.exists(train_py):
        shutil.copy2(train_py, code_dir)
    
    print(f"소스 코드가 {code_dir}에 저장되었습니다.")

def generate_evaluation_results(model, dataloader, device, fold_dir, class_names):
    """모델 평가 및 결과 파일 생성"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="평가 중"):
            coords, labels, _ = batch
            coords = coords.to(device, dtype=torch.float32)
            
            outputs = model(coords, conservative_no_activity=True, apply_transition_rules=True)
            if isinstance(outputs, tuple):
                predictions = outputs[1]  # 후처리된 예측값
            else:
                # 후처리가 적용되지 않은 경우
                outputs = outputs
                _, predictions = torch.max(outputs, dim=-1)
            
            # 배치 차원과 시퀀스 차원을 평평하게
            batch_preds = predictions.detach().cpu().numpy().flatten()
            batch_labels = labels.detach().cpu().numpy().flatten()
            
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
    
    # NumPy 배열로 변환
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 혼동 행렬 계산
    cm = confusion_matrix(all_labels, all_preds)
    
    # 혼동 행렬 저장 (CSV)
    cm_path = os.path.join(fold_dir, "confusion_matrix.csv")
    with open(cm_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + class_names)
        for i, row in enumerate(cm):
            writer.writerow([class_names[i]] + list(row))
    
    # 혼동 행렬 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, "confusion_matrix.png"))
    plt.close()
    
    # 분류 보고서
    report = classification_report(all_labels, all_preds, 
                                  target_names=class_names, 
                                  digits=4)
    
    with open(os.path.join(fold_dir, "classification_report.txt"), 'w') as f:
        f.write(report)
    
    # 클래스별 정확도
    class_acc = {}
    for i, class_name in enumerate(class_names):
        mask = (all_labels == i)
        if np.sum(mask) > 0:  # 해당 클래스의 샘플이 있는 경우만
            acc = np.mean(all_preds[mask] == i) * 100
            class_acc[class_name] = acc
    
    # 클래스별 정확도 저장
    with open(os.path.join(fold_dir, "class_accuracy.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'accuracy'])
        for class_name, acc in class_acc.items():
            writer.writerow([class_name, f"{acc:.2f}%"])
    
    return cm, report, class_acc

def combine_evaluation_results(fold_dirs, combined_results_dir, class_names):
    """모든 폴드의 평가 결과를 결합"""
    # 모든 폴드의 혼동 행렬 합치기
    combined_cm = None
    combined_class_acc = {class_name: [] for class_name in class_names}
    
    for fold_dir in fold_dirs:
        # 혼동 행렬 로드
        cm_path = os.path.join(fold_dir, "confusion_matrix.csv")
        if os.path.exists(cm_path):
            cm = []
            with open(cm_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # 헤더 건너뛰기
                for row in reader:
                    cm.append([int(x) for x in row[1:]])
            cm = np.array(cm)
            
            if combined_cm is None:
                combined_cm = cm
            else:
                combined_cm += cm
        
        # 클래스별 정확도 로드
        acc_path = os.path.join(fold_dir, "class_accuracy.csv")
        if os.path.exists(acc_path):
            with open(acc_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # 헤더 건너뛰기
                for row in reader:
                    class_name, acc = row
                    acc = float(acc.strip('%'))
                    combined_class_acc[class_name].append(acc)
    
    # 결합된 혼동 행렬 저장
    if combined_cm is not None:
        # CSV 저장
        cm_path = os.path.join(combined_results_dir, "combined_confusion_matrix.csv")
        with open(cm_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([''] + class_names)
            for i, row in enumerate(combined_cm):
                writer.writerow([class_names[i]] + list(row))
        
        # 시각화
        plt.figure(figsize=(10, 8))
        sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Combined Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(combined_results_dir, "confusion_matrix.png"))
        plt.close()
    
    # 결합된 클래스별 정확도 계산 및 저장
    class_acc_path = os.path.join(combined_results_dir, "class_accuracy.csv")
    with open(class_acc_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'mean_accuracy', 'std_accuracy'])
        for class_name, accs in combined_class_acc.items():
            if accs:  # 리스트가 비어있지 않은 경우만
                mean_acc = np.mean(accs)
                std_acc = np.std(accs)
                writer.writerow([class_name, f"{mean_acc:.2f}%", f"{std_acc:.2f}%"])
    
    # 결합된 분류 보고서 생성
    if combined_cm is not None:
        # 클래스별 precision, recall, f1-score 계산
        precisions = np.zeros(len(class_names))
        recalls = np.zeros(len(class_names))
        f1_scores = np.zeros(len(class_names))
        
        for i in range(len(class_names)):
            # precision = TP / (TP + FP)
            tp = combined_cm[i, i]
            col_sum = np.sum(combined_cm[:, i])
            precisions[i] = tp / col_sum if col_sum > 0 else 0
            
            # recall = TP / (TP + FN)
            row_sum = np.sum(combined_cm[i, :])
            recalls[i] = tp / row_sum if row_sum > 0 else 0
            
            # f1-score = 2 * (precision * recall) / (precision + recall)
            if precisions[i] + recalls[i] > 0:
                f1_scores[i] = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])
            else:
                f1_scores[i] = 0
        
        # 전체 정확도
        total_samples = np.sum(combined_cm)
        accuracy = np.sum(np.diag(combined_cm)) / total_samples if total_samples > 0 else 0
        
        # 분류 보고서 저장
        report_path = os.path.join(combined_results_dir, "classification_report.txt")
        with open(report_path, 'w') as f:
            f.write("Combined Classification Report\n\n")
            f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-score':<12} {'Support':<12}\n")
            f.write("-" * 60 + "\n")
            
            for i, class_name in enumerate(class_names):
                support = np.sum(combined_cm[i, :])
                f.write(f"{class_name:<15} {precisions[i]:.4f}       {recalls[i]:.4f}       ")
                f.write(f"{f1_scores[i]:.4f}       {support:<12}\n")
            
            f.write("\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Macro Avg: {np.mean(precisions):.4f}    {np.mean(recalls):.4f}    {np.mean(f1_scores):.4f}    {total_samples}\n")
            
            # 가중 평균 계산
            weights = np.sum(combined_cm, axis=1) / total_samples
            w_precision = np.sum(precisions * weights)
            w_recall = np.sum(recalls * weights)
            w_f1 = np.sum(f1_scores * weights)
            f.write(f"Weighted Avg: {w_precision:.4f}    {w_recall:.4f}    {w_f1:.4f}    {total_samples}\n")

def process_predictions(pred_sequence):
    """예측 시퀀스를 JSON 형식으로 변환"""
    # 클래스 인덱스와 이름 매핑
    class_indices = {i: name for i, name in enumerate(class_mapping.values())}
    
    # 결과 JSON 생성
    result = {
        "predictions": [],
        "metadata": {
            "model_type": "TCN",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # 각 프레임에 대한 예측 추가
    for i, pred in enumerate(pred_sequence):
        frame_result = {
            "frame": i,
            "predicted_class": class_indices[pred],
            "class_index": int(pred)
        }
        result["predictions"].append(frame_result)
    
    return result

def test(model, dataloader, device, save_dir=None, dataset_path=None, return_predictions=False):
    """테스트 데이터에 대한 모델 평가 및 예측 결과 생성"""
    model.eval()
    all_preds = []
    all_labels = []
    all_json_results = {}  # JSON 결과 저장용
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="테스트 중"):
            coords, labels, filenames = batch
            coords = coords.to(device, dtype=torch.float32)
            
            # 모델 예측
            outputs = model(coords, conservative_no_activity=True, apply_transition_rules=True)
            
            # 튜플로 반환되는 경우 처리
            if isinstance(outputs, tuple):
                logits, predictions = outputs
            else:
                logits = outputs
                _, predictions = torch.max(logits, dim=-1)
            
            # 파일별 결과 처리
            for i, filename in enumerate(filenames):
                # 예측 결과를 JSON 형식으로 변환
                pred_sequence = predictions[i].cpu().numpy()
                json_result = process_predictions(pred_sequence)
                
                # 결과 저장
                all_json_results[filename] = json_result
                
                # 라벨과 예측값 수집
                if labels is not None:
                    all_preds.extend(predictions[i].cpu().numpy())
                    all_labels.extend(labels[i].cpu().numpy())
    
    # 결과 저장
    if save_dir and dataset_path:
        # output_json 폴더를 다시 생성하지 않고 직접 save_dir에 저장
        json_dir = save_dir
        
        for filename, result in all_json_results.items():
            # 원본 파일 이름에서 .csv 제거하고 _prediction.json 추가
            json_filename = os.path.basename(filename).replace('.csv', '_prediction.json')
            json_path = os.path.join(json_dir, json_filename)
            
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=4)
    
    # 평가 지표 계산
    metrics = {}
    if all_labels and all_preds:
        # 정확도 계산
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
        metrics["accuracy"] = accuracy
        
        # 혼동 행렬
        cm = confusion_matrix(all_labels, all_preds)
        metrics["confusion_matrix"] = cm.tolist()
    
    if return_predictions:
        return metrics, all_json_results
    else:
        return metrics

def train_model(dataset, config):
    device = torch.device(config["device"])
    
    # 결과 저장 디렉토리 생성
    result_dir = create_result_dir()
    combined_results_dir = os.path.join(result_dir, "combined_results")
    os.makedirs(combined_results_dir, exist_ok=True)
    
    # output_json 폴더 생성
    output_json_dir = os.path.join(combined_results_dir, "output_json")
    os.makedirs(output_json_dir, exist_ok=True)
    
    # 소스 코드 저장
    save_source_code(result_dir)
    
    # 하이퍼파라미터 저장
    with open(os.path.join(combined_results_dir, "hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    # K-fold 교차 검증 준비
    kfold = StratifiedKFold(n_splits=config["training"]["num_folds"], shuffle=True, random_state=config["seed"])
    
    # 데이터셋의 레이블 추출
    all_labels = [label[0] for _, label, _ in dataset]  # 첫 프레임의 레이블만 사용
    
    fold_accuracies = []
    fold_losses = []
    fold_results = []
    fold_dirs = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(dataset)), all_labels), 1):
        print(f"\nFold {fold} 시작")
        fold_dir = os.path.join(result_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        fold_dirs.append(fold_dir)
        
        # 데이터로더 생성
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], 
                                sampler=train_sampler)
        valid_loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], 
                                sampler=valid_sampler)
        
        # 모델 초기화
        model = TCN(
            input_size=config["model"]["input_size"],
            output_size=config["model"]["output_size"],
            num_channels=config["model"]["hidden_channels"],
            kernel_size=config["model"]["kernel_size"],
            dropout=config["model"]["dropout"],
            use_se=config["model"]["use_se"]
        ).to(device)
        
        # 손실 함수와 옵티마이저 설정
        loss_params = {"gamma": config["training"]["loss_params"]["gamma"]}
        
        # 클래스 가중치가 있으면 weight_dict로 전달
        if "class_weights" in config["training"]["loss_params"] and config["training"]["loss_params"]["class_weights"] is not None:
            weight_dict = config["training"]["loss_params"]["class_weights"]
            # 클래스 매핑을 사용하여 인덱스 기반 가중치 배열 생성
            class_names = list(class_mapping.values())
            weight_tensor = torch.ones(len(class_names), device=device)
            for i, class_name in enumerate(class_names):
                if class_name in weight_dict:
                    weight_tensor[i] = weight_dict[class_name]
            
            # 가중치 텐서를 weight 매개변수로 전달
            loss_params["weight"] = weight_tensor
        
        criterion = WeightedFocalLoss(**loss_params)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )
        
        # 스케줄러 설정
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config["training"]["scheduler_params"]["T_0"],
            T_mult=config["training"]["scheduler_params"]["T_mult"]
        )
        
        # 모델 학습
        model, best_acc = train_with_early_stopping(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=config["training"]["num_epochs"],
            patience=config["training"]["early_stopping_patience"],
            fold_dir=fold_dir
        )
        
        # 최종 검증 손실 계산
        final_valid_loss, final_valid_acc = validate(model, valid_loader, criterion, device)
        
        # 모델 평가 및 결과 생성 
        generate_evaluation_results(model, valid_loader, device, fold_dir, config["classes"])
        
        fold_accuracies.append(best_acc)
        fold_losses.append(final_valid_loss)
        fold_results.append({
            "fold": fold,
            "loss": final_valid_loss,
            "accuracy": best_acc
        })
        
        print(f"Fold {fold} 완료 - 최고 검증 정확도: {best_acc:.2f}%, 손실: {final_valid_loss:.6f}")
        
        # 모델 저장
        torch.save(model.state_dict(), os.path.join(fold_dir, "final_model.pth"))
    
    # 모든 폴드의 결과 결합
    combine_evaluation_results(fold_dirs, combined_results_dir, config["classes"])
    
    # 전체 결과 계산
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_loss = np.mean(fold_losses)
    std_loss = np.std(fold_losses)
    
    print(f"\n교차 검증 결과:")
    print(f"평균 정확도: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"평균 손실: {mean_loss:.6f} ± {std_loss:.6f}")
    
    # 결과 저장
    final_results = {
        "mean_test_loss": float(mean_loss),
        "std_test_loss": float(std_loss),
        "mean_test_accuracy": float(mean_acc),
        "std_test_accuracy": float(std_acc),
        "fold_results": fold_results
    }
    
    # 업데이트된 config에 results 추가
    config["results"] = final_results
    
    # 최종 하이퍼파라미터와 결과 저장
    with open(os.path.join(combined_results_dir, "hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    # 결과 요약 파일 저장
    with open(os.path.join(result_dir, "results_summary.txt"), "w") as f:
        f.write(f"평균 정확도: {mean_acc:.2f}% ± {std_acc:.2f}%\n")
        f.write(f"평균 손실: {mean_loss:.6f} ± {std_loss:.6f}\n\n")
        f.write("각 폴드 결과:\n")
        for i, (acc, loss) in enumerate(zip(fold_accuracies, fold_losses), 1):
            f.write(f"Fold {i}: 정확도 = {acc:.2f}%, 손실 = {loss:.6f}\n")
    
    print(f"모든 결과가 {result_dir}에 저장되었습니다.")
    
    # 모든 폴드의 모델을 사용하여 테스트 데이터셋에 대한 예측 및 결과 저장
    if hasattr(dataset, 'test_dataset') and dataset.test_dataset:
        print("\n테스트 데이터셋에 대한 평가 시작")
        test_loader = DataLoader(dataset.test_dataset, batch_size=config["training"]["batch_size"])
        
        # 각 폴드의 모델 평가
        for fold, fold_dir in enumerate(fold_dirs, 1):
            model_path = os.path.join(fold_dir, "final_model.pth")
            if os.path.exists(model_path):
                # 모델 로드
                model = TCN(
                    input_size=config["model"]["input_size"],
                    output_size=config["model"]["output_size"],
                    num_channels=config["model"]["hidden_channels"],
                    kernel_size=config["model"]["kernel_size"],
                    dropout=config["model"]["dropout"],
                    use_se=config["model"]["use_se"]
                ).to(device)
                model.load_state_dict(torch.load(model_path))
                
                # 테스트 실행 및 결과 저장
                metrics, json_results = test(
                    model, 
                    test_loader, 
                    device, 
                    save_dir=output_json_dir,  # output_json_dir 바로 전달
                    dataset_path=getattr(dataset.test_dataset, 'csv_dir', None),  # getattr로 안전하게 접근
                    return_predictions=True
                )
                
                # 테스트 결과 저장
                test_results_path = os.path.join(fold_dir, "test_metrics.json")
                with open(test_results_path, 'w') as f:
                    json.dump(metrics, f, indent=4)
                
                print(f"Fold {fold} 테스트 정확도: {metrics.get('accuracy', 'N/A'):.2f}%")
    
    return result_dir, final_results

def save_learning_curves(train_losses, valid_losses, valid_accs, save_path):
    """학습 및 검증 손실과 정확도 내역을 CSV 파일로 저장합니다."""
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'valid_loss', 'valid_accuracy'])
        for epoch, (train_loss, valid_loss, valid_acc) in enumerate(zip(train_losses, valid_losses, valid_accs)):
            writer.writerow([epoch + 1, train_loss, valid_loss, valid_acc]) 
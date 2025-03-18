import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import json
import csv
from tqdm import tqdm
import time
import random
from sklearn.metrics import confusion_matrix, classification_report

# 모듈 import 제거
# from src.models import TCN
# from src.losses import WeightedFocalLoss
# from src.training import train_with_early_stopping, test
# from src.utils import set_seed, ensemble_predictions_with_confidence, apply_temporal_consistency
# from src.datasets import SkeletonDataset

##########################################
# 필요한 클래스 및 함수 정의
##########################################

# 클래스 매핑
class_mapping = {0: "standing", 1: "sitting", 2: "walking", 3: "no_activity", 4: "no_presence"}
activity_to_label = {"standing": 0, "sitting": 1, "walking": 2, "no_activity": 3, "no_presence": 4, "no presence": 4}

def set_seed(seed=42):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

##########################################
# Dataset 정의
##########################################
class SkeletonDataset(Dataset):
    def __init__(self, csv_dir, json_dir, transform=None, training=True, use_augmentation=True):
        self.csv_dir = csv_dir
        self.json_dir = json_dir
        self.transform = transform
        self.training = training
        self.use_augmentation = use_augmentation and training
        self.csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
        
        # 클래스 분포 계산 (스트래티파이드 분할용)
        self.file_labels = self._compute_file_labels()
    
    def _compute_file_labels(self):
        """각 파일의 주요 레이블 계산 (가장 빈번한 레이블)"""
        file_labels = {}
        for idx in range(len(self.csv_files)):
            csv_path = os.path.join(self.csv_dir, self.csv_files[idx])
            df = pd.read_csv(csv_path)
            
            json_path = os.path.join(self.json_dir, self.csv_files[idx].replace(".csv", ".json"))
            with open(json_path, 'r') as f:
                info = json.load(f)
            
            # 가장 긴 활동 세그먼트의 레이블 사용
            main_label = 3  # 기본값
            max_duration = 0
            
            if isinstance(info, list):
                for item in info:
                    if "frameRange" in item and "activity" in item:
                        start_frame, end_frame = item["frameRange"]
                        duration = end_frame - start_frame
                        if duration > max_duration and item["activity"] in activity_to_label:
                            max_duration = duration
                            main_label = activity_to_label[item["activity"]]
            
            file_labels[self.csv_files[idx]] = main_label
        return file_labels
    
    def __len__(self):
        return len(self.csv_files)
    
    def __getitem__(self, idx):
        # SkeletonDataset의 구현...
        # 필요한 코드 추가
        csv_path = os.path.join(self.csv_dir, self.csv_files[idx])
        df = pd.read_csv(csv_path)
        
        if 'frame' in df.columns:
            df = df[df['frame'] != 'frame']
            frames = df['frame'].values.astype(np.int64)
            joint_columns = [col for col in df.columns if col != 'frame']
        else:
            frames = np.arange(len(df))
            joint_columns = df.columns
        
        # 좌표 추출
        coords = df[joint_columns].values.astype(np.float32)
        
        # 데이터 전처리
        coords = self._normalize_coords(coords)
        
        # 증강 코드 추가 필요
        
        coords = torch.tensor(coords, dtype=torch.float32)
        
        json_path = os.path.join(self.json_dir, self.csv_files[idx].replace(".csv", ".json"))
        with open(json_path, 'r') as f:
            info = json.load(f)
        
        max_frame = max(frames) if len(frames) > 0 else len(df)
        labels = np.ones(int(max_frame) + 1, dtype=np.int64) * 3
        
        if isinstance(info, list):
            for item in info:
                if "frameRange" in item and "activity" in item:
                    start_frame, end_frame = item["frameRange"]
                    label = activity_to_label.get(item["activity"], 3)
                    labels[start_frame:end_frame] = label
        
        frame_labels = labels[frames]
        frame_labels = torch.tensor(frame_labels, dtype=torch.long)
        return coords, frame_labels, self.csv_files[idx]
    
    def _normalize_coords(self, coords):
        """좌표 정규화"""
        mean = np.mean(coords, axis=0)
        std = np.std(coords, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return (coords - mean) / std
    
    def get_class_distribution(self):
        """데이터셋의 클래스 분포 반환"""
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for label in self.file_labels.values():
            class_counts[label] += 1
        return class_counts

##########################################
# 손실 함수 정의
##########################################
class WeightedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        # 클래스별 가중치
        self.weights = torch.tensor([3.5, 3.0, 1.2, 1.0, 1.5])
        
    def forward(self, inputs, targets):
        # 기본 cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # focal 가중치 계산
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        # 클래스별 가중치 적용
        if self.weights.device != inputs.device:
            self.weights = self.weights.to(inputs.device)
        
        # 최종 손실 = 클래스 가중치 * focal 가중치 * CE 손실
        loss = self.weights[targets] * focal_weight * ce_loss
        
        # 손실 감소 방식
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

##########################################
# 모델 정의
##########################################
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.3, use_se=False):
        super(TemporalBlock, self).__init__()
        self.use_se = use_se
        
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        if self.use_se:
            self.se = SEBlock(n_outputs)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        # 가중치 초기화
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight.data)
        nn.init.kaiming_normal_(self.conv2.weight.data)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight.data)

    def forward(self, x):
        out = self.net(x)
        if self.use_se:
            out = self.se(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.3, use_se=False):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout, use_se=use_se)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, use_se=False):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout, use_se=use_se)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.attention = nn.Linear(num_channels[-1], 1)  # 어텐션 가중치 계산

    def forward(self, x):
        # x 형태: [batch, time, features]
        # TCN은 [batch, features, time] 형태를 기대
        x = x.transpose(1, 2)
        
        # TCN을 통과
        z = self.tcn(x)
        
        # [batch, channels, time] -> [batch, time, channels]
        z = z.transpose(1, 2)
        
        # 어텐션 메커니즘 적용
        attn_weights = torch.sigmoid(self.attention(z))
        z_weighted = z * attn_weights
        
        # 선형 레이어로 클래스 예측
        y = self.linear(z_weighted)
        
        return y

##########################################
# 훈련, 검증, 테스트 함수
##########################################
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(dataloader, desc="훈련 중"):
        coords, labels, _ = batch
        coords = coords.to(device, dtype=torch.float32)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(coords)
        
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
            
            outputs = model(coords)
            
            B, T, _ = outputs.shape
            outputs = outputs.reshape(B * T, -1)
            labels = labels.reshape(B * T)
            
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return epoch_loss / len(dataloader), 100 * correct / total

def test(model, dataloader, criterion, device, save_dir=None, return_predictions=False, csv_dir=None):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    file_predictions = {}
    file_softmax = {}
    
    # 디렉토리 생성
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch in dataloader:
            # 현재 데이터 형식에 맞게 언패킹
            if len(batch) == 3:
                coords, labels, filenames = batch
                lengths = [coords.size(1)] * coords.size(0)
            else:
                coords, labels, filenames, lengths = batch
                
            coords = coords.to(device, dtype=torch.float32)
            labels = labels.to(device)
            
            # 순전파
            outputs = model(coords)
            
            # 샘플별 처리
            for i, (filename, length) in enumerate(zip(filenames, lengths)):
                # 유효한 범위만 추출
                if isinstance(length, torch.Tensor):
                    length = length.item()
                
                sample_outputs = outputs[i, :length] if length < outputs.size(1) else outputs[i]
                sample_labels = labels[i, :length] if length < labels.size(1) else labels[i]
                
                # 예측 및 소프트맥스 확률
                probs = F.softmax(sample_outputs, dim=1)
                _, preds = torch.max(sample_outputs, dim=1)
                
                # 파일별 예측 저장
                if return_predictions:
                    file_predictions[filename] = preds.cpu().numpy()
                    file_softmax[filename] = probs.cpu().numpy()
                
                # 파일에 결과 저장
                if save_dir and csv_dir:
                    try:
                        # 원본 CSV 파일 읽어서 실제 프레임 수 확인
                        csv_path = os.path.join(csv_dir, filename)
                        df = pd.read_csv(csv_path)
                        num_frames = len(df)
                        
                        # 원본 프레임 수로 제한하여 예측 결과 가공
                        pred_array = preds.cpu().numpy()
                        sample_results = process_predictions(pred_array, max_frame=num_frames-1)
                        
                        # JSON 저장
                        json_path = os.path.join(save_dir, filename.replace(".csv", "_prediction.json"))
                        with open(json_path, 'w') as f:
                            json.dump(sample_results, f, indent=4)
                    except Exception as e:
                        print(f"파일 {filename} 처리 중 오류 발생: {e}")
                
                # 유효한 레이블만 평가에 사용
                valid_mask = (sample_labels >= 0) & (sample_labels < 5)
                if valid_mask.any():
                    valid_outputs = sample_outputs[valid_mask]
                    valid_labels = sample_labels[valid_mask]
                    
                    loss = criterion(valid_outputs, valid_labels)
                    total_loss += loss.item()
                    
                    # 유효한 예측만 저장
                    all_preds.extend(preds[valid_mask].cpu().numpy())
                    all_labels.extend(sample_labels[valid_mask].cpu().numpy())
    
    # 정확도 및 혼동 행렬 계산
    accuracy = 0
    cm = None
    if all_preds:
        accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
        cm = confusion_matrix(all_labels, all_preds, labels=range(5))
    
    # 분류 보고서 저장
    if save_dir and all_preds:
        # 혼동 행렬 저장
        np.savetxt(os.path.join(save_dir, 'confusion_matrix.csv'), cm, delimiter=',', fmt='%d')
        
        # 분류 보고서 저장
        report = classification_report(all_labels, all_preds, 
                                      target_names=list(class_mapping.values()))
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
    
    if return_predictions:
        return total_loss / len(dataloader) if len(dataloader) > 0 else 0, accuracy, cm, file_predictions, file_softmax
    else:
        return total_loss / len(dataloader) if len(dataloader) > 0 else 0, accuracy, cm

def train_with_early_stopping(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, 
                           num_epochs=200, patience=20, fold_dir=None):
    best_valid_acc = 0
    best_model_path = os.path.join(fold_dir, "best_model.pth")
    counter = 0
    
    # 학습 곡선 저장용 변수
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
    
    # 학습 곡선 저장
    if fold_dir:
        save_learning_curves(
            train_loss_history,
            valid_loss_history,
            valid_acc_history,
            os.path.join(fold_dir, "learning_curves.csv")
        )
    
    model.load_state_dict(torch.load(best_model_path))
    return model, best_valid_acc

##########################################
# 유틸리티 함수
##########################################
def save_learning_curves(train_losses, valid_losses, valid_accs, save_path):
    """학습 및 검증 손실과 정확도 내역을 CSV 파일로 저장합니다."""
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'valid_loss', 'valid_accuracy'])
        for epoch, (train_loss, valid_loss, valid_acc) in enumerate(zip(train_losses, valid_losses, valid_accs)):
            writer.writerow([epoch + 1, train_loss, valid_loss, valid_acc])

def process_predictions(predictions, max_frame=None):
    """
    예측 결과를 JSON 형식으로 변환하고 최대 프레임을 제한합니다.
    """
    results = []
    if len(predictions) == 0:
        return results
    
    # 최대 프레임 번호 제한 (지정된 경우)
    if max_frame is not None:
        predictions = predictions[:max_frame+1]  # 0부터 max_frame까지 포함
    
    current_label = predictions[0]
    start_frame = 0
    
    for frame, label in enumerate(predictions):
        if label != current_label:
            results.append({
                "frameRange": [int(start_frame), int(frame)], 
                "activity": class_mapping[int(current_label)]
            })
            current_label = label
            start_frame = frame
    
    # 마지막 세그먼트 추가 (끝 프레임은 마지막 인덱스)
    last_frame = len(predictions)
    results.append({
        "frameRange": [int(start_frame), int(last_frame)], 
        "activity": class_mapping[int(current_label)]
    })
    
    return results

def apply_temporal_consistency(predictions, min_duration=10):
    """시간적 일관성 적용 - 너무 짧은 세그먼트 제거"""
    results = []
    if len(predictions) == 0:
        return results
    
    # 연속된 같은 레이블 세그먼트로 변환
    segments = []
    current_label = predictions[0]
    start_frame = 0
    
    for frame, label in enumerate(predictions):
        if label != current_label:
            segments.append((start_frame, frame, current_label))
            current_label = label
            start_frame = frame
    
    segments.append((start_frame, len(predictions), current_label))
    
    # 너무 짧은 세그먼트 필터링 (앞뒤 세그먼트와 병합)
    filtered_segments = []
    for i, (start, end, label) in enumerate(segments):
        duration = end - start
        if duration >= min_duration:
            filtered_segments.append((start, end, label))
        else:
            # 짧은 세그먼트는 앞뒤 세그먼트 중 더 긴 것으로 병합
            if i > 0 and i < len(segments) - 1:
                prev_duration = segments[i-1][1] - segments[i-1][0]
                next_duration = segments[i+1][1] - segments[i+1][0]
                if prev_duration >= next_duration:
                    if filtered_segments:
                        filtered_segments[-1] = (filtered_segments[-1][0], end, filtered_segments[-1][2])
                else:
                    pass
            elif i > 0:
                if filtered_segments:
                    filtered_segments[-1] = (filtered_segments[-1][0], end, filtered_segments[-1][2])
            elif i < len(segments) - 1:
                pass
    
    for start, end, label in filtered_segments:
        results.append({
            "frameRange": [int(start), int(end)],
            "activity": class_mapping[int(label)]
        })
    
    return results

def ensemble_predictions_with_confidence(all_predictions, softmax_outputs, max_frame=None):
    """신뢰도 기반 앙상블로 더 정확하게 예측 통합하고 최대 프레임 제한"""
    if not all_predictions:
        return []
    
    # 모든 예측 중 가장 긴 것을 기준으로 함
    max_frames = max(len(preds) for preds in all_predictions)
    
    # 최대 프레임 제한이 있으면 적용
    if max_frame is not None:
        max_frames = min(max_frames, max_frame + 1)  # 0부터 max_frame까지 포함
    
    ensembled_preds = np.zeros(max_frames, dtype=np.int64)
    
    # 각 프레임에 대한 클래스별 확률 합산
    frame_probs = np.zeros((max_frames, 5))  # 5개 클래스
    
    for preds, probs in zip(all_predictions, softmax_outputs):
        for frame in range(min(len(preds), len(probs), max_frames)):
            frame_probs[frame] += probs[frame]
    
    # 주변 프레임 확률 평균으로 시간적 일관성 적용 (이동 평균)
    smoothed_probs = np.copy(frame_probs)
    window_size = 5
    
    for frame in range(max_frames):
        start = max(0, frame - window_size//2)
        end = min(max_frames, frame + window_size//2 + 1)
        smoothed_probs[frame] = np.mean(frame_probs[start:end], axis=0)
    
    for frame in range(max_frames):
        ensembled_preds[frame] = np.argmax(smoothed_probs[frame])
    
    return ensembled_preds

def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 장치: {device}")
    
    csv_dir = "data/csv"
    json_dir = "data/json"
    
    # 하이퍼파라미터
    batch_size = 32
    num_epochs = 200
    learning_rate = 0.001
    weight_decay = 0.001
    
    # 모델 파라미터
    input_size = 34
    num_classes = 5
    hidden_channels = [64, 128, 256, 256, 128]
    kernel_size = 5
    dropout = 0.3
    
    dataset = SkeletonDataset(csv_dir=csv_dir, json_dir=json_dir, use_augmentation=True)
    
    num_folds = 5
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    all_test_losses = []
    all_test_accs = []
    all_confusion_matrices = []
    
    ensemble_predictions = {}
    ensemble_softmax = {}
    
    y_labels = np.array(list(dataset.file_labels.values()))
    
    # 기본 결과 저장 폴더를 "tcn_result"로 설정
    base_result_dir = "tcn_result"
    os.makedirs(base_result_dir, exist_ok=True)
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(len(dataset)), y_labels)):
        print(f"\n{'='*30} Fold {fold+1}/{num_folds} {'='*30}")
        # 각 Fold 결과는 tcn_result/fold_x 폴더에 저장
        fold_dir = os.path.join(base_result_dir, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)
        
        train_indices, valid_indices = [], []
        for idx in train_idx:
            if np.random.random() < 0.8:
                train_indices.append(idx)
            else:
                valid_indices.append(idx)
        
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        test_sampler = SubsetRandomSampler(test_idx)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
        
        model = TCN(input_size=input_size, output_size=num_classes, num_channels=hidden_channels,
                    kernel_size=kernel_size, dropout=dropout, use_se=True).to(device)
                    
        criterion = WeightedFocalLoss(gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        model, best_val_acc = train_with_early_stopping(
            model, train_loader, valid_loader, criterion, optimizer, scheduler, device,
            num_epochs=num_epochs, patience=20, fold_dir=fold_dir
        )
        
        test_loss, test_acc, cm, fold_predictions, fold_softmax = test(
            model, test_loader, criterion, device,
            save_dir=os.path.join(fold_dir, "test_results"),
            return_predictions=True,
            csv_dir=csv_dir
        )
        
        for filename, preds in fold_predictions.items():
            if filename not in ensemble_predictions:
                ensemble_predictions[filename] = []
                ensemble_softmax[filename] = []
            ensemble_predictions[filename].append(preds)
            ensemble_softmax[filename].append(fold_softmax[filename])
        
        all_test_losses.append(test_loss)
        all_test_accs.append(test_acc)
        all_confusion_matrices.append(cm)
        
        print(f"Fold {fold+1} - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
    
    # 앙상블 결과 폴더도 tcn_result 하위에 생성
    ensemble_result_dir = os.path.join(base_result_dir, "ensemble_results")
    os.makedirs(ensemble_result_dir, exist_ok=True)
    for filename, all_preds in ensemble_predictions.items():
        ensembled_preds = ensemble_predictions_with_confidence(all_preds, ensemble_softmax[filename])
        final_preds = apply_temporal_consistency(ensembled_preds, min_duration=10)
        json_path = os.path.join(ensemble_result_dir, filename.replace(".csv", "_prediction.json"))
        with open(json_path, 'w') as f:
            json.dump(final_preds, f, indent=4)
    
    print("\n" + "="*60)
    print("5-Fold 교차 검증 결과:")
    print(f"평균 테스트 손실: {np.mean(all_test_losses):.4f} ± {np.std(all_test_losses):.4f}")
    print(f"평균 테스트 정확도: {np.mean(all_test_accs):.2f}% ± {np.std(all_test_accs):.2f}%")
    
    for fold in range(num_folds):
        print(f"Fold {fold+1} - Loss: {all_test_losses[fold]:.4f}, Accuracy: {all_test_accs[fold]:.2f}%")

if __name__ == "__main__":
    main()


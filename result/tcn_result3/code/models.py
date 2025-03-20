import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

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
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, use_se=False):
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
            
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
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
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, use_se=False):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                       dilation=dilation_size, padding=(kernel_size-1)*dilation_size,
                                       dropout=dropout, use_se=use_se)]
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, use_se=False):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout, use_se=use_se)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.attention = nn.Linear(num_channels[-1], 1)
        self.no_activity_threshold = 0.8  # no_activity 클래스(인덱스 3)에 대한 엄격한 threshold
        self.temporal_window = 5  # 시간적 일관성을 위한 윈도우 크기
        
        # CRF 레이어 추가
        self.crf = CRF(output_size, batch_first=True)
        
        # 전이 제약 설정
        self.forbidden_transitions = {
            (1, 2): -1000,  # "sitting -> walking" 금지
            (2, 0): -1000   # "walking -> standing" 금지
        }
        
    def _set_transitions(self):
        """CRF의 전이 매트릭스에 금지된 전이 적용"""
        for (i, j), score in self.forbidden_transitions.items():
            self.crf.transitions.data[i, j] = score
    
    def forward(self, x, conservative_no_activity=True, apply_transition_rules=True):
        # x shape: [batch, time, features]
        x = x.transpose(1, 2)  # TCN은 [batch, features, time] 입력을 기대
        z = self.tcn(x)
        z = z.transpose(1, 2)  # 다시 [batch, time, channels]로 변환
        attn_weights = torch.sigmoid(self.attention(z))
        z_weighted = z * attn_weights
        logits = self.linear(z_weighted)
        
        # 금지된 전이 설정
        if apply_transition_rules:
            self._set_transitions()
        
        batch_size, seq_len, num_classes = logits.shape
        
        if not apply_transition_rules:
            # 규칙 적용 안함 - 단순 argmax
            _, predictions = torch.max(F.softmax(logits, dim=-1), dim=-1)
            return logits
        
        # CRF 디코딩 - 전체 시퀀스를 고려한 최적 경로
        # 태그 시퀀스에 대한 마스크 생성 (모든 시퀀스가 유효하다고 가정)
        mask = torch.ones(batch_size, seq_len, dtype=torch.uint8, device=logits.device)
        
        # Viterbi 디코딩으로 최적 태그 시퀀스 찾기
        predictions = self.crf.decode(logits, mask)
        
        # 텐서로 변환
        predictions_tensor = torch.zeros((batch_size, seq_len), dtype=torch.long, device=logits.device)
        for b, pred_seq in enumerate(predictions):
            for t, p in enumerate(pred_seq):
                if t < seq_len:  # 시퀀스 길이를 초과하지 않도록 체크
                    predictions_tensor[b, t] = p
        
        # no_activity에 대한 보수적 처리
        if conservative_no_activity:
            # 기본적인 확률 계산
            probs = F.softmax(logits, dim=-1)
            
            # 현재 예측에서 no_activity 클래스(인덱스 3)에 대한 보수적 처리
            for b in range(batch_size):
                for t in range(seq_len):
                    # no_activity로 예측되었지만 확률이 threshold보다 낮은 경우
                    if predictions_tensor[b, t] == 3 and probs[b, t, 3] < self.no_activity_threshold:
                        # 주변 프레임 확인
                        start = max(0, t - self.temporal_window // 2)
                        end = min(seq_len, t + self.temporal_window // 2 + 1)
                        
                        # 윈도우 내에서 no_activity가 아닌 클래스가 충분히 많은지 확인
                        window = predictions_tensor[b, start:end]
                        non_no_activity = (window != 3).sum().item()
                        
                        # 윈도우의 50% 이상이 no_activity가 아니면 no_activity 예측 취소
                        if non_no_activity > (end-start) * 0.5:
                            # 3(no_activity)가 아닌 클래스 중 가장 높은 확률을 가진 클래스로 변경
                            probs_except_no_activity = probs[b, t].clone()
                            probs_except_no_activity[3] = 0  # no_activity 확률을 0으로 설정
                            _, new_pred = torch.max(probs_except_no_activity, dim=0)
                            predictions_tensor[b, t] = new_pred
        
        return logits, predictions_tensor.long()
    
    def loss(self, logits, labels, mask=None):
        """CRF 손실 계산 함수"""
        if mask is None:
            batch_size, seq_len = labels.shape
            mask = torch.ones(batch_size, seq_len, dtype=torch.uint8, device=logits.device)
        return -self.crf(logits, labels, mask=mask, reduction='mean')

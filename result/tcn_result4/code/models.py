import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.no_activity_threshold = 0.9  # no_activity 클래스(인덱스 3)에 대한 엄격한 threshold
        self.temporal_window = 5  # 시간적 일관성을 위한 윈도우 크기
        
        # 전이 규칙을 위한 전이 매트릭스 (FSM)
        # 행: 현재 상태, 열: 다음 상태
        # 0: standing, 1: sitting, 2: walking, 3: no_activity, 4: no_presence
        # 값 0은 전이 불가, 1은 전이 가능
        self.transition_matrix = torch.ones(5, 5)
        # "sitting -> walking" 금지
        self.transition_matrix[1, 2] = 0
        # "walking -> standing" 금지
        self.transition_matrix[2, 0] = 0
        
    def forward(self, x, conservative_no_activity=True, apply_transition_rules=True):
        # x shape: [batch, time, features]
        x = x.transpose(1, 2)  # TCN은 [batch, features, time] 입력을 기대
        z = self.tcn(x)
        z = z.transpose(1, 2)  # 다시 [batch, time, channels]로 변환
        attn_weights = torch.sigmoid(self.attention(z))
        z_weighted = z * attn_weights
        logits = self.linear(z_weighted)
        
        # 기본적으로 가장 높은 확률의 클래스로 예측
        probs = F.softmax(logits, dim=-1)
        batch_size, seq_len, num_classes = probs.shape
        _, predictions = torch.max(probs, dim=-1)
        
        # conservative_no_activity 또는 apply_transition_rules 중 하나라도 True면 후처리 적용
        if conservative_no_activity or apply_transition_rules:
            # 후처리된 예측을 저장할 텐서
            processed_preds = predictions.clone()
            
            # no_activity 클래스에 대한 보수적 측정 적용
            if conservative_no_activity:
                # no_activity 클래스(인덱스 3)에 대한 엄격한 threshold 적용
                no_activity_mask = (probs[:, :, 3] >= self.no_activity_threshold)
                
                # 시간적 일관성 적용 (이동 평균)
                for b in range(batch_size):
                    for t in range(seq_len):
                        # 현재 위치 주변 윈도우
                        start = max(0, t - self.temporal_window // 2)
                        end = min(seq_len, t + self.temporal_window // 2 + 1)
                        
                        # 윈도우 내에서 no_activity가 아닌 클래스가 충분히 많으면 no_activity 예측 취소
                        window = predictions[b, start:end]
                        non_no_activity = (window != 3).sum().item()
                        
                        # 윈도우의 50% 이상이 no_activity가 아니면, no_activity 예측을 취소
                        if predictions[b, t] == 3 and non_no_activity > (end-start) * 0.5:
                            # 3(no_activity)가 아닌 클래스 중 가장 높은 확률을 가진 클래스로 변경
                            probs_except_no_activity = probs[b, t].clone()
                            probs_except_no_activity[3] = 0  # no_activity 확률을 0으로 설정
                            _, new_pred = torch.max(probs_except_no_activity, dim=0)
                            processed_preds[b, t] = new_pred
            
            # 전이 규칙 적용
            if apply_transition_rules and seq_len > 1:
                for b in range(batch_size):
                    for t in range(1, seq_len):
                        prev_state = processed_preds[b, t-1].item()
                        curr_state = processed_preds[b, t].item()
                        
                        # 현재 전이가 유효하지 않은 경우 처리
                        if self.transition_matrix[prev_state, curr_state] == 0:
                            # 현재 상태의 확률 분포에서, 유효한 전이만 남기고 나머지는 0으로 설정
                            valid_probs = probs[b, t].clone()
                            
                            # 전이 규칙에 따라 불가능한 상태로의 확률을 0으로 설정
                            for next_state in range(num_classes):
                                if self.transition_matrix[prev_state, next_state] == 0:
                                    valid_probs[next_state] = 0
                            
                            # 만약 모든 확률이 0이 되면(가능한 전이가 없으면) 이전 상태 유지
                            if valid_probs.sum() == 0:
                                processed_preds[b, t] = processed_preds[b, t-1]
                            else:
                                # 유효한 전이 중 가장 확률이 높은 것 선택
                                _, new_pred = torch.max(valid_probs, dim=0)
                                processed_preds[b, t] = new_pred
            
            return logits, processed_preds.long()
        
        return logits

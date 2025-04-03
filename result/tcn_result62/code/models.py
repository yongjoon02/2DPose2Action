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
        self.no_activity_threshold = 0.98  # 더 엄격한 threshold
        self.temporal_window = 15  # 시간적 일관성 강화
        
        # 전이 매트릭스 강화
        self.transition_matrix = torch.ones(5, 5)
        # 기본 전이 규칙
        self.transition_matrix[1, 2] = 0  # sitting -> walking 금지
        self.transition_matrix[2, 0] = 0  # walking -> standing 금지
        
        # standing/sitting과 no_activity 간의 전이를 더 엄격하게 제한
        self.transition_matrix[0, 3] = 0.5  # standing -> no_activity (매우 제한)
        self.transition_matrix[1, 3] = 0.5  # sitting -> no_activity (매우 제한)
        self.transition_matrix[3, 0] = 1.5  # no_activity -> standing (더 쉽게 허용)
        self.transition_matrix[3, 1] = 1.5  # no_activity -> sitting (더 쉽게 허용)
        
        # walking 관련 전이 조정
        self.transition_matrix[2, 3] = 0.7  # walking -> no_activity
        self.transition_matrix[3, 2] = 0.7  # no_activity -> walking
        
    def forward(self, x, conservative_no_activity=True, apply_transition_rules=True):
        # x shape: [batch, time, features]
        x = x.transpose(1, 2)
        z = self.tcn(x)
        z = z.transpose(1, 2)
        attn_weights = torch.sigmoid(self.attention(z))
        z_weighted = z * attn_weights
        logits = self.linear(z_weighted)
        
        probs = F.softmax(logits, dim=-1)
        batch_size, seq_len, num_classes = probs.shape
        _, predictions = torch.max(probs, dim=-1)
        
        if conservative_no_activity or apply_transition_rules:
            processed_preds = predictions.clone()
            
            if conservative_no_activity:
                for b in range(batch_size):
                    for t in range(seq_len):
                        # 현재 위치 주변 윈도우
                        start = max(0, t - self.temporal_window // 2)
                        end = min(seq_len, t + self.temporal_window // 2 + 1)
                        window = predictions[b, start:end]
                        
                        # 현재 예측이 no_activity(3)인 경우
                        if predictions[b, t] == 3:
                            # 윈도우 내 활동 비율 계산
                            activity_ratio = (window != 3).float().mean()
                            
                            # 주변에 충분한 활동이 있으면 no_activity 취소
                            if activity_ratio > 0.4:  # 임계값 조정 (0.45 -> 0.4)
                                # 이전과 이후 프레임의 예측을 고려하여 새로운 레이블 결정
                                if t > 0 and t < seq_len - 1:
                                    prev_pred = predictions[b, t-1]
                                    next_pred = predictions[b, t+1]
                                    
                                    # 이전/이후 프레임이 같은 활동이면 그 활동으로 변경
                                    if prev_pred == next_pred and prev_pred != 3:
                                        processed_preds[b, t] = prev_pred
                                    else:
                                        # 아니면 주변 프레임에서 가장 빈번한 활동으로 변경
                                        window_activities = window[window != 3]
                                        if len(window_activities) > 0:
                                            mode_activity = torch.mode(window_activities)[0]
                                            processed_preds[b, t] = mode_activity
            
            if apply_transition_rules and seq_len > 1:
                for b in range(batch_size):
                    for t in range(1, seq_len):
                        prev_state = processed_preds[b, t-1].item()
                        curr_state = processed_preds[b, t].item()
                        
                        if self.transition_matrix[prev_state, curr_state] < 1:
                            # 현재 상태의 확률 분포에서, 유효한 전이만 고려
                            valid_probs = probs[b, t].clone()
                            
                            for next_state in range(num_classes):
                                if self.transition_matrix[prev_state, next_state] < 1:
                                    valid_probs[next_state] = 0
                            
                            if valid_probs.sum() == 0:
                                processed_preds[b, t] = processed_preds[b, t-1]
                            else:
                                _, new_pred = torch.max(valid_probs, dim=0)
                                processed_preds[b, t] = new_pred
            
            return logits, processed_preds.long()
        
        return logits
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
        
        # 정적 상태에서 no_activity로의 전환에 대한 더 엄격한 임계값
        self.static_to_no_activity_threshold = 0.95  # 더 높은 임계값
        self.no_activity_to_static_threshold = 0.7   # 낮은 임계값 (쉽게 돌아올 수 있도록)
        self.consecutive_frames_threshold = 3        # 연속 프레임 임계값
        
        # 클래스별 상태 지속 시간 정보
        self.state_durations = {
            3: 20,  # no_activity -  2초이상 지속
        }
        
        # 전이 규칙을 위한 전이 매트릭스 (FSM)
        # 행: 현재 상태, 열: 다음 상태
        # 0: standing, 1: sitting, 2: walking, 3: no_activity, 4: no_presence
        # 값 0은 전이 불가, 1은 전이 가능
        self.transition_matrix = torch.ones(5, 5)
        # "sitting -> walking" 금지
        self.transition_matrix[1, 2] = 0
        # "walking -> standing" 금지
        self.transition_matrix[2, 0] = 0
        # 정적 상태(standing, sitting)와 no_activity 간의 직접 전환을 제한
        # 이 값들은 0으로 설정하지 않고 낮은 값으로 설정하여 매우 높은 확신이 있을 때만 전환 허용
        self.transition_matrix[0, 3] = 0.3  # standing -> no_activity (낮은 확률로 허용)
        self.transition_matrix[1, 3] = 0.3  # sitting -> no_activity (낮은 확률로 허용)
        self.transition_matrix[3, 0] = 0.7  # no_activity -> standing (비교적 쉽게 허용)
        self.transition_matrix[3, 1] = 0.7  # no_activity -> sitting (비교적 쉽게 허용)
        
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
            
            # no_activity 클래스의 지속 시간 추적용 카운터
            state_durations = {3: 0}  # no_activity만 추적
            
            # no_activity 클래스에 대한 보수적 측정 적용
            if conservative_no_activity:
                # 정적 상태별 보수적 측정을 위한 마스크 생성
                static_states = torch.tensor([0, 1], device=x.device)  # standing, sitting
                
                for b in range(batch_size):
                    # no_activity 연속 프레임 카운터
                    consecutive_frames_no_activity = 0
                    prev_pred = None
                    
                    for t in range(seq_len):
                        current_pred = predictions[b, t].item()
                        
                        # 이전 예측과 같으면서 no_activity인 경우 연속 카운트 증가
                        if prev_pred == 3 and current_pred == 3:
                            consecutive_frames_no_activity += 1
                        elif current_pred == 3:
                            consecutive_frames_no_activity = 1
                        else:
                            consecutive_frames_no_activity = 0
                        
                        prev_pred = current_pred
                        
                        # 정적 상태(standing, sitting)에서 no_activity로의 전환 처리
                        if current_pred == 3 and t > 0 and processed_preds[b, t-1].item() in [0, 1]:
                            # 더 엄격한 임계값 적용
                            if probs[b, t, 3] < self.static_to_no_activity_threshold:
                                # 이전 상태의 확률이 일정 수준 이상이면 이전 상태 유지
                                prev_state = processed_preds[b, t-1].item()
                                if probs[b, t, prev_state] > 0.2:  # 20% 이상 확률이면 이전 상태 유지
                                    processed_preds[b, t] = processed_preds[b, t-1]
                                else:
                                    # 아니면 no_activity 이외의 가장 높은 확률 클래스 선택
                                    temp_probs = probs[b, t].clone()
                                    temp_probs[3] = 0  # no_activity 확률 제거
                                    _, new_pred = torch.max(temp_probs, dim=0)
                                    processed_preds[b, t] = new_pred
                        
                        # no_activity에서 정적 상태로의 전환은 더 쉽게 허용
                        elif current_pred in [0, 1] and t > 0 and processed_preds[b, t-1].item() == 3:
                            # 정적 상태로의 전환 확률이 임계값보다 높으면 허용
                            if probs[b, t, current_pred] >= self.no_activity_to_static_threshold:
                                processed_preds[b, t] = torch.tensor(current_pred, device=x.device)
                        
                        # 시간적 일관성 적용 (이동 평균) - 특히 정적 상태 중요
                        if current_pred == 3:  # no_activity 예측인 경우
                            # 현재 위치 주변 윈도우
                            start = max(0, t - self.temporal_window // 2)
                            end = min(seq_len, t + self.temporal_window // 2 + 1)
                            
                            # 윈도우 내에서 정적 상태(standing, sitting)의 비율 확인
                            window = predictions[b, start:end]
                            static_count = ((window == 0) | (window == 1)).sum().item()
                            
                            # 윈도우의 40% 이상이 정적 상태이면, no_activity 예측을 취소하고 주요 정적 상태로 변경
                            if static_count > (end-start) * 0.4:
                                # 윈도우 내에서 가장 빈번한 정적 상태 파악
                                standing_count = (window == 0).sum().item()
                                sitting_count = (window == 1).sum().item()
                                
                                if standing_count >= sitting_count:
                                    new_pred = 0  # standing
                                else:
                                    new_pred = 1  # sitting
                                    
                                processed_preds[b, t] = torch.tensor(new_pred, device=x.device)
                            else:
                                # 최소 지속 시간 조건 만족 여부 확인
                                # no_activity 예측이 연속 3프레임 미만이면 이전 상태로 되돌림
                                if consecutive_frames_no_activity < self.consecutive_frames_threshold and t > 0:
                                    processed_preds[b, t] = processed_preds[b, t-1]
            
            # 전이 규칙 적용
            if apply_transition_rules and seq_len > 1:
                for b in range(batch_size):
                    for t in range(seq_len):
                        curr_state = processed_preds[b, t].item()
                        
                        # 첫 프레임이 아닌 경우 전이 규칙 적용
                        if t > 0:
                            prev_state = processed_preds[b, t-1].item()
                            
                            # 상태가 변경된 경우
                            if curr_state != prev_state:
                                # no_activity에서 다른 상태로 전환될 때만 지속 시간 확인
                                if prev_state == 3:
                                    if state_durations[3] < self.state_durations[3]:
                                        # no_activity의 최소 지속 시간에 도달하지 못한 경우 이전 상태 유지
                                        processed_preds[b, t] = torch.tensor(prev_state, device=x.device)
                                        curr_state = prev_state  # 상태 업데이트
                                else:
                                    # 전이 행렬 확인
                                    transition_prob = self.transition_matrix[prev_state, curr_state].item()
                                    
                                    # 전이 확률이 낮은 경우 (0이 아닌 낮은 값)
                                    if 0 < transition_prob < 1:
                                        # 현재 상태의 확률이 전이 임계값을 넘지 못하면 이전 상태 유지
                                        if probs[b, t, curr_state] < transition_prob:
                                            processed_preds[b, t] = torch.tensor(prev_state, device=x.device)
                                            curr_state = prev_state  # 상태 업데이트
                                    # 전이가 불가능한 경우 (0)
                                    elif transition_prob == 0:
                                        # 유효한 전이 찾기
                                        valid_probs = probs[b, t].clone()
                                        for next_state in range(num_classes):
                                            if self.transition_matrix[prev_state, next_state] == 0:
                                                valid_probs[next_state] = 0
                                        
                                        # 유효한 전이가 있으면 가장 확률이 높은 것 선택
                                        if valid_probs.sum() > 0:
                                            _, new_pred = torch.max(valid_probs, dim=0)
                                            processed_preds[b, t] = new_pred
                                            curr_state = new_pred.item()  # 상태 업데이트
                                        else:
                                            # 유효한 전이가 없으면 이전 상태 유지
                                            processed_preds[b, t] = torch.tensor(prev_state, device=x.device)
                                            curr_state = prev_state  # 상태 업데이트
                        
                        # 현재 상태가 no_activity인 경우만 지속 시간 추적
                        if curr_state == 3:
                            state_durations[3] += 1
                        elif prev_state == 3 and curr_state != 3:
                            state_durations[3] = 0
            
            return logits, processed_preds.long()
        
        return logits

import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        # 클래스별 가중치 (standing, sitting, walking, no_activity, no_presence)
        self.weights = torch.tensor([3.4, 3.6, 1.2, 0.7, 1.5])
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        if self.weights.device != inputs.device:
            self.weights = self.weights.to(inputs.device)
        loss = self.weights[targets] * focal_weight * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()
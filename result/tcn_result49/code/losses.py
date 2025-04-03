import torch.nn as nn

class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, inputs, targets):
        return self.criterion(inputs, targets)
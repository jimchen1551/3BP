import torch
import torch.nn as nn
import torch.nn.functional as F

class BalancedBCELoss(nn.Module):
    def __init__(self):
        super(BalancedBCELoss, self).__init__()

    def forward(self, inputs, targets):
        beta = torch.sum(targets, dim=1) / targets.shape[1]
        x = torch.clamp(torch.log(inputs), min=-100)
        y = torch.clamp(torch.log(1 - inputs), min=-100)
        l = -(beta.unsqueeze(1) * targets * x + (1 - beta).unsqueeze(1) * (1 - targets) * y)
        loss = torch.sum(l)
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.view(-1, 1))
        logpt = logpt.view(-1)
        pt = logpt.exp()
        focal_loss = -1 * (1 - pt) ** self.gamma * logpt
        balanced_focal_loss = self.alpha * focal_loss
        return balanced_focal_loss.mean()

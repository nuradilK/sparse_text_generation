import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.functional import relu
import torch.nn.functional as F

class _GenericLoss(nn.Module):
    def __init__(self, ignore_index=-100, reduction="elementwise_mean"):
        assert reduction in ["elementwise_mean", "sum", "none"]
        self.reduction = reduction
        self.ignore_index = ignore_index
        super(_GenericLoss, self).__init__()

    def forward(self, X, target):
        loss = self.loss(X, target)
        if self.ignore_index >= 0:
            ignored_positions = target == self.ignore_index
            size = float((target.size(0) - ignored_positions.sum()).item())
            loss.masked_fill_(ignored_positions, 0.0)
        else:
            size = float(target.size(0))
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "elementwise_mean":
            loss = loss.sum() / size
        return loss
    
class ReluLossFunction(nn.Module):
    def __init__(self):
        super(ReluLossFunction, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
    
    def forward(self, X, target):
        """
        X (FloatTensor): n x num_classes
        target (LongTensor): n, the indices of the target classes
        """
        assert X.shape[0] == target.shape[0]
        
        y_hat = F.relu(X)
        loss = 0.5 * (self.mse(X, F.one_hot(target, num_classes=X.shape[1]).float().cuda()) - self.mse(X, y_hat)) / X.shape[0]
        
        return loss

class ReluLoss(_GenericLoss):
    def __init__(self, ignore_index=-100, reduction="elementwise_mean"):
        super(ReluLoss, self).__init__(ignore_index, reduction)
        self.loss_func = ReluLossFunction()

    def loss(self, X, target):
        return self.loss_func(X, target)
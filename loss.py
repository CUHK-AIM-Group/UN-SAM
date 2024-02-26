import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.classification import MulticlassConfusionMatrix
from pytorch_lightning.metrics import ConfusionMatrix
from pytorch_lightning.metrics.classification import F1

class BinaryMaskLoss(nn.Module):
    def __init__(self, weight=0.8, size_average=True):
        super(BinaryMaskLoss, self).__init__()

        self.weight = weight

    def forward(self, inputs, targets, smooth=1):

        # inputs = inputs.view(-1)
        # targets = targets.view(-1)

        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        dice_score = 1-dice

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = 0.8 * (1 - BCE_EXP)**2 * BCE
                       
        return self.weight * dice_score + (1 - self.weight) * focal_loss
        

class BinaryIoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BinaryIoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
                     
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth) / (union + smooth)
                
        return IoU


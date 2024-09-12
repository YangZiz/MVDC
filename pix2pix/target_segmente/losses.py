import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import one_hot_3d


#损失函数
class cross_entropy_2D(nn.Module):
    def __init__(self,weight=None, reduction=True):
        super().__init__()
        self.weight = weight
        self.reduction = reduction


    def forward(self,input,target):
        n, c, h, w = input.size()
        log_p = F.log_softmax(input, dim = 1)
        log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
        target = target.view(target.numel())
        loss = F.nll_loss(log_p, target, weight = self.weight, reduction='sum')
        if self.reduction:
            loss /= float(target.numel())

        return loss



# class DiceLoss(nn.Module):
#
#     def __init__(self,num_classes,weights = None):
#         super().__init__()
#         self.num_classes = num_classes
#         # self.ce = cross_entropy_3D(weight = weights)
#
#     def forward(self, pred, target):
#
#
#         # ce_loss = self.ce(pred,target)
#
#         #one hot
#         n,_, s, h = target.size()
#         one_hot = torch.zeros(n, self.num_classes, s, h).cuda()
#         target = one_hot.scatter_(1, target.view(n, 1, s, h), 1)
#
#
#         pred = torch.softmax(pred,1)
#         smooth = 1
#         dice = 0.
#         # dice系数的定义
#         for i in range(pred.size(1)):
#             dice += 2 * (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / \
#                     (pred[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
#                                                 target[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
#         # 返回的是dice距离
#         dice = dice / pred.size(1)
#         loss = torch.clamp((1 - dice).mean(), 0, 1)
#         return loss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc
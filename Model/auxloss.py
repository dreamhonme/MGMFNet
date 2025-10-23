import torch.nn as nn
import torch.nn.functional as F
class SpatialConstraintLoss(nn.Module):
    def __init__(self):
        super(SpatialConstraintLoss, self).__init__()

    def forward(self, pred, target):
        # 计算预测和目标的梯度
        pred_grad_x = pred[:, :, :-1, :] - pred[:, :, 1:, :]  # 水平梯度
        pred_grad_y = pred[:, :, :, :-1] - pred[:, :, :, 1:]  # 垂直梯度

        target_grad_x = target[:, :, :-1, :] - target[:, :, 1:, :]
        target_grad_y = target[:, :, :, :-1] - target[:, :, :, 1:]

        # 计算梯度差异的损失
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)

        return loss_x + loss_y
class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1,ws = 1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.spatial = SpatialConstraintLoss()  # 空间约束损失模块
        self.wb = wb
        self.wd = wd
        self.ws = ws  # 空间约束损失的权重
    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)
        spatial_loss = self.spatial(pred, target)  # 计算空间约束损失
        total_loss = self.wb * bceloss + self.wd * diceloss + self.ws * spatial_loss
        return total_loss
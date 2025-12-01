import torch
import torch.nn as nn

class AsymmetricLoss(nn.Module):
    """
    非对称损失函数 (Asymmetric Loss)
    参考论文: "Asymmetric Loss For Multi-Label Classification"
    
    Args:
        gamma_neg: 负样本的聚焦参数 (默认4)
        gamma_pos: 正样本的聚焦参数 (默认1)
        clip: 概率裁剪阈值，用于hard thresholding (默认0.05)
        eps: 数值稳定性参数
        disable_torch_grad_focal_loss: 是否禁用focal loss的梯度计算
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, x, y):
        """
        Args:
            x: 模型输出 (已经过sigmoid)
            y: 真实标签
        """
        # 计算正负样本的概率
        x_sigmoid = x  # 已经过sigmoid
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping (硬阈值裁剪)
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # 基础交叉熵损失
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing (非对称聚焦)
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            
            loss *= one_sided_w

        return -loss.mean()


class AsymmetricLossOptimized(nn.Module):
    """
    优化版本的非对称损失函数，计算更高效
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

        # 预防死亡神经元
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        self.targets = y
        self.anti_targets = 1 - y

        # 计算概率
        self.xs_pos = x
        self.xs_neg = 1.0 - self.xs_pos

        # 非对称裁剪
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # 基础CE计算
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # 非对称聚焦
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            
            self.loss *= self.asymmetric_w

        return -self.loss.mean()
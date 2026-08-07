import torch
import torch.nn.functional as F


def bce_with_logits(logits, targets):
    """
    对 F.binary_cross_entropy_with_logits 的简单封装，
    方便统一调用并确保 reduction='mean'
    """
    return F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')


def nnpu_loss(logits_p, logits_u, pi: float):
    """
    Non-negative PU Loss (nnPU)

    Args:
        logits_p: 正例集合的 logits 输出 (未经过 sigmoid)
        logits_u: 未标记集合的 logits 输出 (未经过 sigmoid)
        pi: 正例先验概率 P(y=1)

    Returns:
        risk: 计算出的风险值 (Scalar)
    """
    # 避免空 Batch 导致 NaN 或报错
    if logits_p.numel() == 0 or logits_u.numel() == 0:
        # 返回一个带梯度的 0，防止计算图中断
        return torch.tensor(0.0, device=logits_p.device, requires_grad=True)

    # 计算各个部分的风险项
    # R_p+ = (1/Np) * sum(l(f(x), 1))
    # 即：正例被预测为正例的损失
    loss_pos = F.binary_cross_entropy_with_logits(logits_p, torch.ones_like(logits_p), reduction='mean')

    # R_u- = (1/Nu) * sum(l(f(x), 0))
    # 即：未标记数据被预测为负例的损失
    loss_neg_on_u = F.binary_cross_entropy_with_logits(logits_u, torch.zeros_like(logits_u), reduction='mean')

    # R_p- = (1/Np) * sum(l(f(x), 0))
    # 即：正例被预测为负例的损失 (用于修正 U 中的潜在正例)
    loss_neg_on_p = F.binary_cross_entropy_with_logits(logits_p, torch.zeros_like(logits_p), reduction='mean')

    # nnPU Risk 公式:
    # Risk = pi * R_p+ + max(0, R_u- - pi * R_p-)
    # 这里的 pi 对应论文中的 class prior
    risk = pi * loss_pos + torch.clamp(loss_neg_on_u - pi * loss_neg_on_p, min=0.0)

    return risk


def taylor_cross_entropy(logits, targets, n=2, reduction='mean'):
    """
    Taylor Expansion Loss (t-CE)
    L = \sum (1 - p)^k / k
    相比 BCE，它对从 U 中被误标记为 N 的样本（False Negatives）容忍度更高。
    """
    probs = torch.sigmoid(logits)
    # 对于正例 target=1: loss = \sum (1-p)^k / k
    # 对于负例 target=0: loss = \sum (p)^k / k

    p_t = torch.where(targets == 1, probs, 1 - probs)
    loss = 0
    for k in range(1, n + 1):
        loss += (1 - p_t) ** k / k

    if reduction == 'mean':
        return loss.mean()
    return loss


def taylor_pu_loss(logits_p, logits_u, prior_pi=None):
    """
    Simplified Taylor Variational PU Loss
    核心思想：在 PU 设定下，使用 Taylor Loss 替代普通 BCE，
    可以减少对未知正例（U中隐含的P）的过度惩罚。

    注意：纯正的 VPU 需要极复杂的变分界推导。
    这里我们使用 Taylor Loss 的鲁棒性特性作为 'Variational Strategy' 的一种实现。
    """
    # 1. 正例项 (Positive Risk)
    # 使用 Taylor Loss 计算正例损失
    loss_p = taylor_cross_entropy(logits_p, torch.ones_like(logits_p), n=2)

    # 2. 未标记项 (Unlabeled Risk)
    # 我们希望 U 被判为负，但给予它“泰勒级数”的平滑，防止过度确信
    loss_u = taylor_cross_entropy(logits_u, torch.zeros_like(logits_u), n=2)

    # 3. 组合 (这里不需要显式的 pi，或者 pi 的影响被泰勒项平滑了)
    # 这种 Loss 对 Label Noise 不敏感
    return loss_p + loss_u
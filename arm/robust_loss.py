import torch

class RobustLossAgent:
    def __init__(self, margin=0.05):
        self.margin = margin

    def robust_loss(self, delta_dis, y, robust_loss, robust_parameter=None, epoch=0):
        robust_parameter = robust_parameter or {}

        if robust_loss == 'reweight':
            beta = robust_parameter.get("beta", 0.2)
            return self.reweighted_triplet_loss(delta_dis, y, beta=beta)

        elif robust_loss == 'truncated':
            alpha = robust_parameter.get("alpha", 0.05)
            epsilon_max = robust_parameter.get("epsilon_max", 0.2)
            return self.truncated_triplet_loss(delta_dis, y, epoch, alpha=alpha, epsilon_max=epsilon_max)
        
        elif robust_loss == 'label_smoothing':
            alpha = robust_parameter.get("alpha", 0.05)
            return self.label_smoothing_triplet_loss(delta_dis, y, alpha=alpha)

        else:
            raise ValueError(f"unknown robust method: {robust_loss}")

    def reweighted_triplet_loss(self, delta_dis, y, beta=0.2, eps=1e-6):
        pred = torch.sigmoid(delta_dis)
        weight = pred ** beta
        raw_loss = weight * torch.clamp(self.margin - y * delta_dis, min=0.0)
        normalized_loss = raw_loss.sum() * (delta_dis.size(0) / (weight.sum() + eps))
        return normalized_loss

    def truncated_triplet_loss(self, delta_dis, y, epoch, alpha=0.1, epsilon_max=0.1):
        loss_sample = torch.clamp(self.margin - y * delta_dis, min=0.0)
        drop_rate = min(alpha * epoch, epsilon_max)
        batch_size = loss_sample.size(0)

        if drop_rate > 0:
            kth = int(batch_size * (1 - drop_rate))
            if kth <= 0:
                loss_sample = torch.zeros_like(loss_sample)
            else:
                tau = torch.kthvalue(loss_sample, kth).values
                keep_mask = (loss_sample <= tau)
                loss_sample = loss_sample * keep_mask.float()

        return loss_sample.mean()


    def smooth_labels(self, y, alpha, num_classes=2):
        # 将 {-1, +1} 转为 {0, 1}
        y_idx = ((y + 1) / 2).long()

        # 构建 smearing matrix M = (1 - α)I + (α / L) J
        I = torch.eye(num_classes, device=y.device)
        J = torch.ones((num_classes, num_classes), device=y.device)
        M = (1 - alpha) * I + (alpha / num_classes) * J  # shape: (L, L)

        # 使用 M 查表构造 soft label
        soft_labels = M[y_idx]  # shape: (batch_size, num_classes)
        return soft_labels

    def label_smoothing_triplet_loss(self, delta_dis, y, alpha):
        # 构造 soft_label 分布
        soft_labels = self.smooth_labels(y, alpha=alpha, num_classes=2)  # shape: (B, 2)
        
        # 获取 soft 正负权重
        pos_weight = soft_labels[:, 1]  # 对应标签 +1 的 soft 概率
        neg_weight = soft_labels[:, 0]  # 对应标签 -1 的 soft 概率

        # 相当于 y ∈ [-1, +1] 的 soft 版本
        y_smooth = pos_weight - neg_weight  # shape: (B,)

        loss = torch.clamp(self.margin - y_smooth * delta_dis, min=0.0)
        return loss.mean()

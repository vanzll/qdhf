import torch
import torch.nn as nn

class RobustLossAgent:
    def __init__(self, margin=0.05):
        self.margin = margin

    def robust_loss(self, delta_dis, y, robust_loss, parameter, robust_parameter=None, epoch=0):
        robust_parameter = robust_parameter or {}

        if robust_loss == 'reweight':
            beta = 0.2
            return self.reweighted_triplet_loss(delta_dis, y, beta=beta)

        elif robust_loss == 'truncated':
            alpha = 0.05
            epsilon_max = 0.2
            return self.truncated_triplet_loss(delta_dis, y, epoch, alpha=alpha, epsilon_max=epsilon_max)
        
        elif robust_loss == 'label_smoothing':
            alpha = 0.05
            return self.label_smoothing_triplet_loss(delta_dis, y, alpha=alpha)
        
        elif robust_loss == 'rDPO':
            epsilon = parameter
            return self.rDPO_triplet_loss(delta_dis, y, epsilon=epsilon)
        
        elif robust_loss == 'cDPO':
            epsilon = parameter
            return self.cDPO_triplet_loss(delta_dis, y, epsilon=epsilon)
        
        elif robust_loss == 'None':
            loss_fn = lambda y, delta_dis: torch.max(torch.tensor([0.0], device=y.device), 0.05 - y * delta_dis).mean()
            return loss_fn(y, delta_dis)
        
        elif robust_loss == 'robust_qdhf':
            # beta = 0.2
            # return self.reweighted_triplet_loss(delta_dis, y, beta=beta)
            loss_fn = lambda y, delta_dis: torch.max(torch.tensor([0.0], device=y.device), 0.05 - y * delta_dis).mean()
            return loss_fn(y, delta_dis)
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


    def smooth_labels(self, y, alpha=0.1, num_classes=2):
        if alpha <= 1e-6:
            return y, y  # 不平滑

        y_idx = ((y + 1) / 2).long()
        I = torch.eye(num_classes, device=y.device, dtype=y.dtype)
        J = torch.ones((num_classes, num_classes), device=y.device, dtype=y.dtype)
        M = (1 - alpha) * I + (alpha / num_classes) * J

        soft_labels = M[y_idx]
        y_smooth = soft_labels[:, 1] - soft_labels[:, 0]  # 转换为 ∈ [-1, 1]
        y_smooth = torch.clamp(y_smooth, -1.0, 1.0)
        return soft_labels, y_smooth


    def label_smoothing_triplet_loss(self, delta_dis, y, alpha):
        _, y_smooth = self.smooth_labels(y, alpha=alpha, num_classes=2)
        loss = torch.clamp(self.margin - y_smooth * delta_dis, min=0.0)
        return loss.mean()

    
    def rDPO_triplet_loss(self, delta_dis, y, epsilon=0.1):
        # print(epsilon)
        y = y.to(delta_dis.device)
        loss_clean   = -torch.log(torch.sigmoid(delta_dis * y) + 1e-8)     # as if label = +1
        loss_flipped = -torch.log(torch.sigmoid(-delta_dis * y) + 1e-8)    # as if label = -1

        numerator   = (1 - epsilon) * loss_clean - epsilon * loss_flipped
        denominator = 1 - 2 * epsilon
        corrected_loss = numerator / (denominator + 1e-8)  # add epsilon to avoid div 0
        return corrected_loss.mean()
    
    def cDPO_triplet_loss(self, delta_dis, y, epsilon=0.1):
        y = y.to(delta_dis.device)
        log_prob_clean   = torch.log(torch.sigmoid(delta_dis * y) + 1e-8)     # as if label = +1
        log_prob_flipped = torch.log(torch.sigmoid(-delta_dis * y) + 1e-8)    # as if label = -1

        # Weighted sum per noisy label distribution
        loss = -((1 - epsilon) * log_prob_clean + epsilon * log_prob_flipped)
        return loss.mean()



class TripletCDRP(nn.Module):
    def __init__(self, gamma=5.0, eta=0.5, eps=0.01):
        super(TripletCDRP, self).__init__()
        self.gamma = gamma
        self.eta = eta
        self.eps = eps

    def forward(self, ref_embed, x1_embed, x2_embed, pref_label):
        d1 = torch.sum((ref_embed - x1_embed) ** 2, dim=-1)
        d2 = torch.sum((ref_embed - x2_embed) ** 2, dim=-1)  
        logits = torch.stack([-d1, -d2], dim=1)  # (B, 2)
        logits = torch.clamp(logits, min=-20, max=20)
        probs = torch.softmax(logits, dim=1)  # (B, 2)

        labels = ((pref_label + 1) // 2).long()

        # Base CE loss
        ce_loss = nn.NLLLoss(reduction="none")
        log_probs = torch.log(probs + 1e-8)
        # nominal_loss = ce_loss(log_probs, labels)  # (B,)

        # C(y', y) = gamma if y' != y else 0
        gamma = self.gamma
        penalty = gamma * (1 - torch.nn.functional.one_hot(labels, num_classes=2).to(logits.device))

        # robust loss inner: max_y' [l(y') - gamma * 1_{y' ≠ y}]
        loss_all = ce_loss(log_probs.repeat(1, 2).reshape(-1, 2), torch.tensor([0, 1] * logits.shape[0]).to(logits.device)).reshape(-1, 2)
        robust_loss = (loss_all - penalty).max(dim=1).values

        final_loss = self.eps * gamma + robust_loss.mean()

        return final_loss


class InstanceEarlyStopper:
    def __init__(self, num_samples, patience=3, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.loss_history = [[] for _ in range(num_samples)]
        self.mask = torch.zeros(num_samples, dtype=torch.bool)  # True 表示“停止训练该样本”

    def update(self, indices, losses):
        for i, idx in enumerate(indices):
            idx = int(idx)
            if self.mask[idx]:
                continue
            self.loss_history[idx].append(losses[i].item())

            if len(self.loss_history[idx]) >= self.patience:
                l_hist = self.loss_history[idx][-self.patience:]
                if len(l_hist) == self.patience:
                    delta2 = sum([-l_hist[j] + 2*l_hist[j+1] - l_hist[j+2] for j in range(self.patience - 2)])
                    
                    if abs(delta2) < self.delta:
                        self.mask[idx] = True

    def get_active_indices(self, all_indices):
        return [idx for idx in all_indices if not self.mask[int(idx)]]

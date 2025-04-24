import torch

class NoiseGenerator:
    def __init__(self):
        pass

    def generate_noise(self, gt, gt_dis, noisy_method, parameter):
        if noisy_method == 'stochastic':
            return self.stochastic_labels(gt, gt_dis, parameter)
        elif noisy_method == 'noisy_labels_exact':
            return self.noisy_labels_exact(gt, parameter)
        elif noisy_method == 'add_equal_noise':
            return self.add_equal_noise(gt, gt_dis, parameter)
        elif noisy_method == 'flip_by_distance':
            return self.flip_by_distance(gt, gt_dis, parameter)
        elif noisy_method == 'flip_labels_asymmetric':
            return self.flip_labels_asymmetric(gt, parameter)
        else:
            raise ValueError(f"Unknown noisy_method: {noisy_method}")

    def noisy_labels_exact(self, gt, noise_rate):
        n = len(gt)
        n_flip = int(n * noise_rate)
        flip_indices = torch.randperm(n)[:n_flip]
        gt_noisy = gt.clone()
        gt_noisy[flip_indices] = -gt_noisy[flip_indices]
        return gt_noisy

    def stochastic_labels(self, gt, gt_dis, noise_ratio):
        gt = gt.clone()
        n = len(gt)
        n_noise = int(n * noise_ratio)

        noise_indices = torch.randperm(n)[:n_noise]

        gt_dis_tensor = torch.tensor(gt_dis, dtype=torch.float32)
        prob = torch.sigmoid(-gt_dis_tensor[noise_indices])
        sampled = torch.bernoulli(prob)
        noisy_labels = sampled * 2 - 1 

        gt[noise_indices] = noisy_labels
        return gt

    def add_equal_noise(self, gt, gt_dis, delta_equal):
        gt_dis_tensor = torch.tensor(gt_dis, dtype=torch.float32)
        equal_mask = torch.abs(gt_dis_tensor) < delta_equal

        gt_noisy = gt.clone()
        gt_noisy[equal_mask] = 0  
        return gt_noisy

    def flip_by_distance(self, gt, gt_dis, delta_threshold):
        gt_dis_tensor = torch.tensor(gt_dis, dtype=torch.float32)
        flip_mask = torch.abs(gt_dis_tensor) < delta_threshold

        gt_noisy = gt.clone()
        gt_noisy[flip_mask] = -gt_noisy[flip_mask] 
        return gt_noisy

    def flip_labels_asymmetric(self, gt, flip_rate=0.1):
        flip_rate_neg = flip_rate
        flip_rate_pos = flip_rate
        gt = gt.clone()
        pos_indices = torch.where(gt == 1)[0]
        neg_indices = torch.where(gt == -1)[0]

        n_pos_flip = int(len(pos_indices) * flip_rate_pos)
        n_neg_flip = int(len(neg_indices) * flip_rate_neg)

        pos_flip_idx = pos_indices[torch.randperm(len(pos_indices))[
            :n_pos_flip]]
        neg_flip_idx = neg_indices[torch.randperm(len(neg_indices))[
            :n_neg_flip]]

        gt[pos_flip_idx] = -1
        gt[neg_flip_idx] = 1
        return gt

from sklearn.decomposition import PCA
import torch
import copy
from torch import nn
import numpy as np
import time
from generate_noise import NoiseGenerator
from robust_loss import RobustLossAgent, TripletCDRP, InstanceEarlyStopper, NaPOLoss
from evaluate import replace_sample
import matplotlib.pyplot as plt
import os

def plot_multiple_curves(
    lists,          # list of lists，eg: [list1, list2, list3]
    labels,         # list of str，自定义每条线的名字
    save_path,      # str，保存路径
    title="Noise Rate over Epochs",
    xlabel="Epoch",
    ylabel="Noise Rate"
):

    plt.figure(figsize=(8,6))

    num_lists = len(lists)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # 够多线就循环用

    epochs = list(range(1, len(lists[0]) + 1))

    for i in range(num_lists):
        plt.plot(
            epochs,
            lists[i],
            label=labels[i],
            color=colors[i % len(colors)],
            marker='o'
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 自动创建目录
    plt.savefig(save_path)
    plt.close()
    print(f"compete: {save_path}")
    
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_signed_delta_with_noise_breakdown(
    signed_deltas: list,
    is_clean_flags: list,      # list of 0 (noise) or 1 (clean)
    save_path: str,
    bins: int = 100
):

    # 把数据转换成 numpy
    signed_deltas = np.array(signed_deltas)
    is_clean_flags = np.array(is_clean_flags)

    # 分 bin
    bin_counts_total, bin_edges = np.histogram(signed_deltas, bins=bins)
    bin_width = bin_edges[1] - bin_edges[0]

    # 初始化 clean/noise bin counts
    bin_counts_clean = np.zeros_like(bin_counts_total)
    bin_counts_noise = np.zeros_like(bin_counts_total)

    # 分 bin 累加 clean / noise 数
    for i in range(len(signed_deltas)):
        val = signed_deltas[i]
        is_clean = is_clean_flags[i]

        # 找到这个样本落在哪个 bin
        bin_idx = np.searchsorted(bin_edges, val, side='right') - 1
        if 0 <= bin_idx < bins:
            if is_clean:
                bin_counts_clean[bin_idx] += 1
            else:
                bin_counts_noise[bin_idx] += 1

    # 绘图
    x = bin_edges[:-1]  # 每个 bin 的起点
    plt.figure(figsize=(10,6))
    plt.bar(x, bin_counts_clean, width=bin_width, color='skyblue', label='Clean', align='edge')
    plt.bar(x, bin_counts_noise, width=bin_width, bottom=bin_counts_clean, color='salmon', label='Noisy', align='edge')

    plt.xlabel("Signed Delta (y × Δ)")
    plt.ylabel("Sample Count")
    plt.title("Distribution of Signed Delta with Noise Breakdown")
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 图已保存到: {save_path}")


class DisEmbed(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=input_dim * 2, out_features=latent_dim),
        )

    def forward(self, x):
        x = torch.cumsum(x, 1)
        x1 = torch.cos(x)
        x2 = torch.sin(x)
        x = torch.cat([x1, x2], -1)
        x = self.enc(x)
        # x = torch.nn.functional.normalize(x, p=2, dim=1)  # l2 normalize
        return x

    def calc_dis(self, x1, x2):
        x1 = self.forward(x1)
        x2 = self.forward(x2)
        # return 1 - torch.sum(x1 * x2, -1)
        return torch.sum(torch.square(x1 - x2), -1)

    def triplet_delta_dis(self, ref, x1, x2):
        x1 = self.forward(x1)
        x2 = self.forward(x2)
        ref = self.forward(ref)
        # return torch.sum(ref * x2, -1) - torch.sum(ref * x1, -1)
        return torch.sum(torch.square(ref - x1), -1) - torch.sum(
            torch.square(ref - x2), -1
        )


def fit_dis_embed(
    inputs, gt_measures, latent_dim, batch_size=32, seed=None, device="cpu", 
    noisy_method=None, parameter=None, robust_loss=None, itr=1, all_sols=None, log_dir=None
):
    # 这个函数使用 triplet-based contrastive learning，训练一个模型在嵌入空间中表示“人类感知下的多样性”
    # inputs是随机产生的角度
    # print(device)
    # print(inputs.shape) (250, 3, 10)
    # print(gt_measures.shape) (250, 3, 2)
    itr = (itr-1)/100
    t = time.time()
    model = DisEmbed(input_dim=inputs.shape[-1], latent_dim=latent_dim)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = lambda y, delta_dis: torch.max(
        torch.tensor([0.0]), 0.05 - y * delta_dis
    ).mean()  # hinge triplet loss
    n_pref_data = inputs.shape[0]
    ref = inputs[:, 0]
    x1 = inputs[:, 1]
    x2 = inputs[:, 2]

    n_train = int(n_pref_data * 0.75)
    n_val = n_pref_data - n_train

    ref_train = ref[:n_train]
    x1_train = x1[:n_train]
    x2_train = x2[:n_train]
    ref_val = ref[n_train:]
    x1_val = x1[n_train:]
    x2_val = x2[n_train:]
    # ref_val = ref_train
    # x1_val = x1_train
    # x2_val = x2_train

    n_iters_per_epoch = max((n_train) // batch_size, 1)

    ref_gt_measures = gt_measures[:, 0]
    x1_gt_measures = gt_measures[:, 1]
    x2_gt_measures = gt_measures[:, 2]
    ref_gt_train = ref_gt_measures[:n_train]
    x1_gt_train = x1_gt_measures[:n_train]
    x2_gt_train = x2_gt_measures[:n_train]
    ref_gt_val = ref_gt_measures[n_train:]
    x1_gt_val = x1_gt_measures[n_train:]
    x2_gt_val = x2_gt_measures[n_train:]

    gt_dis_all = np.sum(
        (np.square(ref_gt_train - x1_gt_train) - np.square(ref_gt_train - x2_gt_train)),
        axis=-1
    )
    gt_all = torch.tensor(gt_dis_all > 0, dtype=torch.float32) * 2 - 1 
    noise_gen = NoiseGenerator()
    gt_noise_all = noise_gen.generate_noise(
        gt_all, gt_dis_all, noisy_method=noisy_method, parameter=parameter
    )
    val_acc = []
    origin_list, flip_list, resample_list = [],[],[]
    all_signed_deltas, is_clean_flags = [], []
    for epoch in range(1000):
        origin_noise = 0.0
        flip_noise = 0.0
        resample_noise = 0.0
        for _ in range(n_iters_per_epoch):
            idx = np.random.choice(n_train, batch_size)
            idx_torch = torch.tensor(idx, dtype=torch.long) 
            batch_ref = torch.tensor(ref_train[idx], dtype=torch.float32).to(device)
            batch1 = torch.tensor(x1_train[idx], dtype=torch.float32).to(device)
            batch2 = torch.tensor(x2_train[idx], dtype=torch.float32).to(device)
            batch_gt_noise = gt_noise_all[idx].to(device) # .clone().detach()
            batch_gt = gt_all[idx].to(device)
            origin_noise += (batch_gt == batch_gt_noise).sum().item()
            # print(origin_noise)

            if robust_loss == 'crdo':
                optimizer.zero_grad()
                ref_embed = model.forward(batch_ref).to(device)
                x1_embed  = model.forward(batch1).to(device)
                x2_embed  = model.forward(batch2).to(device)
                triplet_loss_fn = TripletCDRP(gamma=5.0, eta=0.5, eps=0.01).to(device)
                loss = triplet_loss_fn(ref_embed, x1_embed, x2_embed, batch_gt_noise).to(device)
            elif robust_loss == 'robust_qdhf':
                delta_dis = model.triplet_delta_dis(batch_ref, batch1, batch2).detach()
                signed_delta = delta_dis * batch_gt_noise
                all_signed_deltas.extend(signed_delta.detach().cpu().tolist())
                is_clean_flags.extend((batch_gt == batch_gt_noise).cpu().tolist())
                neg_mask = signed_delta < 0
                signed_delta_neg = signed_delta[neg_mask]
                if signed_delta_neg.numel() > 0:
                    mean_neg = signed_delta_neg.mean()
                    std_neg = signed_delta_neg.std()
                    threshold1 = mean_neg - 2 * std_neg
                    threshold2 = mean_neg - 0 * std_neg
                    flip_mask = signed_delta <= threshold1
                    replace_mask = (signed_delta > threshold1) & (signed_delta < threshold2)
                    # keep_mask = signed_delta >= threshold2
                    batch_gt_noise[flip_mask] = -batch_gt_noise[flip_mask]
                    flip_noise += (batch_gt == batch_gt_noise).sum().item()
                    for i in torch.where(replace_mask)[0]:
                        batch1[i], batch2[i], batch_gt_noise[i], batch_gt[i] = replace_sample(
                                batch_ref[i], batch1[i], batch2[i],
                                model=model,
                                device=device,
                                K=2,
                                noisy_method=noisy_method,
                                parameter=parameter,
                                all_sols=all_sols,
                                itr=itr
                            )
                # x1_train[idx_torch] = batch1.detach().cpu()
                # x2_train[idx_torch] = batch2.detach().cpu()
                # gt_noise_all[idx_torch] = batch_gt_noise.detach().cpu()
                # gt_all[idx_torch] = batch_gt.detach().cpu()

                resample_noise += (batch_gt == batch_gt_noise).sum().item()
                # print(f"after resample: {count_same.item()}")
                optimizer.zero_grad()
                delta_dis = model.triplet_delta_dis(batch_ref, batch1, batch2)
                loss_agent = RobustLossAgent(margin=0.05)
                loss = loss_agent.robust_loss(delta_dis, batch_gt_noise, robust_loss, parameter, epoch, device).to(device)
            elif robust_loss == "NaPO":
                optimizer.zero_grad()
                loss_fn = NaPOLoss(alpha=0.05)
                delta_dis = model.triplet_delta_dis(batch_ref, batch1, batch2)
                loss = loss_fn(delta_dis, batch_gt_noise)
            else:
                optimizer.zero_grad()
                loss_agent = RobustLossAgent(margin=0.05)
                delta_dis = model.triplet_delta_dis(batch_ref, batch1, batch2)
                loss = loss_agent.robust_loss(delta_dis, batch_gt_noise, robust_loss, parameter, epoch, device).to(device)
            loss.backward()
            optimizer.step()
        if robust_loss == 'robust_qdhf':
            total_sample = batch_size * n_iters_per_epoch
            origin_list.append(1 - origin_noise / total_sample)  # float / int → float，OK
            flip_list.append(1-flip_noise/total_sample)
            resample_list.append(1-resample_noise/total_sample)
            # all_losses.extend(loss.detach().cpu().tolist())



        n_correct = 0
        n_total = 0
        with torch.no_grad():
            idx = np.arange(n_val)
            batch_ref = torch.tensor(ref_val[idx], dtype=torch.float32).to(device)
            batch1 = torch.tensor(x1_val[idx], dtype=torch.float32).to(device)
            batch2 = torch.tensor(x2_val[idx], dtype=torch.float32).to(device)
            delta_dis = model.triplet_delta_dis(batch_ref, batch1, batch2)
            pred = delta_dis > 0
            gt_dis = np.sum(
                (
                    np.square(ref_gt_val[idx] - x1_gt_val[idx])
                    - np.square(ref_gt_val[idx] - x2_gt_val[idx])
                ),
                -1,
            )
            gt = torch.tensor(gt_dis > 0).to(device)
            n_correct += (pred == gt).sum().item()
            n_total += len(idx)

        acc = n_correct / n_total
        val_acc.append(acc)

        if epoch > 10 and np.mean(val_acc[-10:]) < np.mean(val_acc[-11:-1]):
            break
        
    if robust_loss == "robust_qdhf":
        plot_multiple_curves(
            lists=[origin_list, flip_list, resample_list],
            labels=["Origin Noise Rate", "Flip Rate", "Resample Rate"],
            save_path=os.path.join(log_dir, f"noise_rate{itr}.png")
        )
        plot_signed_delta_with_noise_breakdown(
            signed_deltas=all_signed_deltas,
            is_clean_flags=is_clean_flags,
            save_path=os.path.join(log_dir, f"signed_delta_noise_breakdown{itr}.png")
        )

    print(
        f"{np.round(time.time()- t, 1)}s ({epoch} epochs) | DisEmbed (n={n_pref_data}) fitted with val acc.: {acc}"
    )

    return model, acc


class AE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=latent_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=input_dim),
        )

    def forward(self, x):
        return self.enc(x)

    def reconstruct(self, x):
        return self.dec(self.enc(x))


def fit_ae(inputs, latent_dim=2, batch_size=32, device="cpu"):
    model = AE(input_dim=inputs.shape[1], latent_dim=latent_dim)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    n_data = inputs.shape[0]
    n_train = int(n_data * 0.75)
    n_iter_per_epoch = max(n_train // batch_size, 1)

    epoch = 0
    val_loss = []
    while True:
        for _ in range(n_iter_per_epoch):
            idx = np.random.choice(n_train, batch_size)
            batch = torch.tensor(inputs[idx], dtype=torch.float32).to(device)

            optimizer.zero_grad()
            outputs = model.reconstruct(batch)
            loss = loss_fn(outputs, batch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            idx = np.arange(n_train, n_data)
            batch = torch.tensor(inputs[idx], dtype=torch.float32).to(device)
            outputs = model.reconstruct(batch)
            val_loss.append(loss_fn(outputs, batch).item())

        epoch += 1
        if epoch > 10 and np.mean(val_loss[-10:]) > np.mean(val_loss[-11:-1]):
            break

    print(
        f"{epoch} epochs | AE fitted with reconstruction loss: {np.mean(val_loss[-10:])}"
    )
    return model


def fit_pca(features, n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(features)
    return pca

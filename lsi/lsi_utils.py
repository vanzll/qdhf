import time

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn
from generate_noise import NoiseGenerator
from robust_loss import RobustLossAgent, TripletCDRP, InstanceEarlyStopper

class DisEmbed(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=latent_dim),
        )

    def forward(self, x):
        x = self.enc(x)
        return x

    def calc_dis(self, x1, x2):
        x1 = self.forward(x1)
        x2 = self.forward(x2)
        return torch.sum(torch.square(x1 - x2), -1)

    def triplet_delta_dis(self, ref, x1, x2):
        x1 = self.forward(x1)
        x2 = self.forward(x2)
        ref = self.forward(ref)
        return torch.sum(torch.square(ref - x1), -1) - torch.sum(
            torch.square(ref - x2), -1
        )


def fit_dis_embed(
    inputs, gt_measures, latent_dim, batch_size=32, seed=None, device="cpu", noisy_method=None, parameter=None, robust_loss=None
):
    t = time.time()
    model = DisEmbed(input_dim=inputs.shape[-1], latent_dim=latent_dim)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = lambda y, delta_dis: torch.max(
        torch.tensor([0.0]).to(device), 0.05 - y * delta_dis
    ).mean()
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

    val_acc = []
    for epoch in range(1000):
        if epoch < 100:
            early_stopper = InstanceEarlyStopper(num_samples=n_train, patience=3, delta=1e-1)
        else:
            early_stopper = InstanceEarlyStopper(num_samples=n_train, patience=3, delta=5e-2)
        if robust_loss == 'ies':
            full_idx = np.arange(n_train)
            active_idx = early_stopper.get_active_indices(full_idx)

            if len(active_idx) <= batch_size:
                print("All triplets stopped early.")
                break

            batch_idx = np.random.choice(active_idx, batch_size)

            batch_ref = torch.tensor(ref_train[batch_idx], dtype=torch.float32).to(device)
            batch1 = torch.tensor(x1_train[batch_idx], dtype=torch.float32).to(device)
            batch2 = torch.tensor(x2_train[batch_idx], dtype=torch.float32).to(device)

            ref_embed = model.forward(batch_ref)
            x1_embed = model.forward(batch1)
            x2_embed = model.forward(batch2)

            gt_dis = np.sum(
                np.square(ref_gt_train[batch_idx] - x1_gt_train[batch_idx]) -
                np.square(ref_gt_train[batch_idx] - x2_gt_train[batch_idx]),
                axis=-1
            )
            gt = torch.tensor(gt_dis > 0, dtype=torch.float32) * 2 - 1
            noise_gen = NoiseGenerator()
            gt_noise = noise_gen.generate_noise(gt, gt_dis, noisy_method=noisy_method, parameter=parameter).to(device)

            delta_dis = model.triplet_delta_dis(batch_ref, batch1, batch2)
            loss_samplewise = torch.clamp(0.05 - gt_noise * delta_dis, min=0.0)  # shape: (B,)
            
            # if epoch <= 3:
            #     print(f"Mean: {loss_samplewise.mean().item()}")
            #     print(f"Standard Deviation: {loss_samplewise.std().item()}")
            #     print(f"Max: {loss_samplewise.max().item()}")
            #     print(f"Min: {loss_samplewise.min().item()}")

            loss = loss_samplewise.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新 IES 模块
            early_stopper.update(batch_idx, loss_samplewise.detach().cpu())
        else:
            for _ in range(n_iters_per_epoch):
                idx = np.random.choice(n_train, batch_size)
                batch_ref = torch.tensor(ref_train[idx], dtype=torch.float32).to(device)
                batch1 = torch.tensor(x1_train[idx], dtype=torch.float32).to(device)
                batch2 = torch.tensor(x2_train[idx], dtype=torch.float32).to(device)

                optimizer.zero_grad()
                
                gt_dis = np.sum(
                    (
                        np.square(ref_gt_train[idx] - x1_gt_train[idx])
                        - np.square(ref_gt_train[idx] - x2_gt_train[idx])
                    ),
                    -1,
                )
                # print("abs(gt_dis) percentiles:",np.percentile(np.abs(gt_dis), [0, 5, 10, 20, 30, 50, 60, 70, 90, 100]))
                noise_gen = NoiseGenerator()
                gt = torch.tensor(gt_dis > 0, dtype=torch.float32) * 2 - 1 # 产生无噪声标签
                gt_noise = noise_gen.generate_noise(
                    gt, gt_dis, noisy_method=noisy_method, parameter=parameter)
                
                # loss = loss_fn(gt_noise, delta_dis)
                
                # 使用 Reweighted loss
                if robust_loss == 'crdo':
                    ref_embed = model.forward(batch_ref).to(device)
                    x1_embed  = model.forward(batch1).to(device)
                    x2_embed  = model.forward(batch2).to(device)
                    triplet_loss_fn = TripletCDRP(gamma=5.0, eta=0.5, eps=0.01).to(device)
                    loss = triplet_loss_fn(ref_embed, x1_embed, x2_embed, gt_noise).to(device)
                else:
                    loss_agent = RobustLossAgent(margin=0.05)
                    delta_dis = model.triplet_delta_dis(batch_ref, batch1, batch2)
                    loss = loss_agent.robust_loss(delta_dis, gt_noise, robust_loss, parameter, epoch, device).to(device)
                loss.backward()
                optimizer.step()

        # Evaluate.
        n_correct = 0
        n_total = 0
        with torch.no_grad():
            idx = np.arange(n_val)
            batch_ref = ref_val[idx].float()
            batch1 = x1_val[idx].float()
            batch2 = x2_val[idx].float()
            delta_dis = model.triplet_delta_dis(batch_ref, batch1, batch2)
            pred = delta_dis > 0
            gt_dis = torch.nn.functional.cosine_similarity(
                ref_gt_val[idx], x2_gt_val[idx], dim=-1
            ) - torch.nn.functional.cosine_similarity(
                ref_gt_val[idx], x1_gt_val[idx], dim=-1
            )
            gt = torch.tensor(gt_dis > 0)
            n_correct += (pred == gt).sum().item()
            n_total += len(idx)

        acc = n_correct / n_total
        val_acc.append(acc)

        if epoch > 10 and np.mean(val_acc[-10:]) < np.mean(val_acc[-11:-1]):
            break

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


def calc_pairwise_dis(embeddings):
    # embeddings: (n_data, n_features)
    n_data = embeddings.shape[0]
    # cosine distance
    dis = 1 - np.dot(embeddings, embeddings.T) / (
        np.linalg.norm(embeddings, axis=1, keepdims=True)
        * np.linalg.norm(embeddings, axis=1, keepdims=True).T
    )

    mask = (1 - np.eye(n_data)).astype(bool)
    mean_dis = np.mean(dis[mask])
    std_dis = np.std(dis[mask])

    return mean_dis, std_dis

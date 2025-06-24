import numpy as np
import torch
from generate_noise import NoiseGenerator

def evaluate_grasp(
    joint_angles, method, metadata=None, device="cpu", return_features=False
):
    # metadata传入模型用来产生measure，还有其他的附加信息
    objs = -np.var(joint_angles, axis=1)
    # Remap the objective from [-1, 0] to [0, 100]
    objs = (objs + 1.0) * 100.0

    if metadata is None:
        metadata = {}

    cum_theta = np.cumsum(joint_angles, axis=1)
    x_pos = np.cos(cum_theta)
    y_pos = np.sin(cum_theta)
    features = np.concatenate((x_pos, y_pos), axis=1)

    if "dis_embed" in metadata:
        if metadata["dis_embed"] is not None:
            with torch.no_grad():
                features = (
                    metadata["dis_embed"](
                        torch.tensor(joint_angles, dtype=torch.float32).to(device)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

    if method is None:
        if return_features:
            return objs, features
        else:
            return objs
    elif method in ["qd", "gthf"]:
        link_lengths = np.ones(joint_angles.shape[1])
        # theta_1, theta_1 + theta_2, ...
        cum_theta = np.cumsum(joint_angles, axis=1)
        # l_1 * cos(theta_1), l_2 * cos(theta_1 + theta_2), ...
        x_pos = link_lengths[None] * np.cos(cum_theta)
        # l_1 * sin(theta_1), l_2 * sin(theta_1 + theta_2), ...
        y_pos = link_lengths[None] * np.sin(cum_theta)

        if method == "qd":  # 相当于只保留一个二维向量乘以数量，表示“这个 grasp 抓到了哪里”
            measures = np.concatenate(
                (
                    np.sum(x_pos, axis=1, keepdims=True),
                    np.sum(y_pos, axis=1, keepdims=True),
                ),
                axis=1,
            )
        elif method == "gthf": # 记录了所有关节的位置
            measures = np.concatenate(
                (
                    np.cumsum(x_pos, axis=1),
                    np.cumsum(y_pos, axis=1),
                ),
                axis=1,
            )
    elif method == "pca":
        assert "pca" in metadata
        measures = metadata["pca"].transform(features)
    elif method == "ae":
        assert "ae" in metadata
        with torch.no_grad():
            measures = (
                metadata["ae"](torch.tensor(features, dtype=torch.float32).to(device))
                .detach()
                .cpu()
                .numpy()
            )
    elif method == "qdhf":
        measures = features
    else:
        raise NotImplementedError(f"Unknown method: {method}")

    if return_features:
        return objs, measures, features
    else:
        return objs, measures
    

def replace_sample(
    batch_ref_i, batch1_i, batch2_i, model, 
    device="cpu", K=3, noisy_method=None, parameter=None, itr=0, all_sols=None
):
    # ---------- 1. 批量生成 K 个新 p* 与 K 个新 n* ----------
    if itr == 0:
        p_expand = torch.tensor(
            np.random.uniform(low=-np.pi, high=np.pi, size=(K, 10)),
            dtype=torch.float32,
            device=device
        )

        n_expand = torch.tensor(
            np.random.uniform(low=-np.pi, high=np.pi, size=(K, 10)),
            dtype=torch.float32,
            device=device
        )
    else:
        if isinstance(all_sols, np.ndarray):
            # all_sols 是 numpy
            idx_p = np.random.choice(all_sols.shape[0], size=K, replace=False)
            idx_n = np.random.choice(all_sols.shape[0], size=K, replace=False)
            p_expand = torch.tensor(all_sols[idx_p], dtype=torch.float32, device=device)
            n_expand = torch.tensor(all_sols[idx_n], dtype=torch.float32, device=device)
        elif isinstance(all_sols, torch.Tensor):
            # all_sols 已经是 torch.Tensor
            idx_p = torch.randint(0, all_sols.size(0), size=(K,), device=all_sols.device)
            idx_n = torch.randint(0, all_sols.size(0), size=(K,), device=all_sols.device)
            p_expand = all_sols[idx_p].to(device).float()  # 保证 float32
            n_expand = all_sols[idx_n].to(device).float()
        else:
            raise TypeError(f"Unsupported type for all_sols: {type(all_sols)}")

    # ---------- 2. 组装 2K 个 (r,p*,n_old) / (r,p_old,n*) ----------
    r_tile_p = batch_ref_i.unsqueeze(0).repeat(K, 1)      # (K,10)
    r_tile_n = r_tile_p.clone()

    part1 = torch.cat([r_tile_p, p_expand, batch2_i.unsqueeze(0).repeat(K,1)], dim=1)
    part2 = torch.cat([r_tile_n, batch1_i.unsqueeze(0).repeat(K,1), n_expand], dim=1)

    new_batch = torch.cat([part1, part2], dim=0)          # (2K, 30)

    # ---------- 3. 计算 ground-truth measure (NumPy) ----------
    # evaluate_grasp 需要形状 (B, 3, 10)；我们这里 batch=2K
    new_batch_flat = new_batch.view(2 * K, 3, 10).reshape(6 * K, 10)
    new_batch_np   = new_batch_flat.cpu().numpy().astype(np.float32)
    # print(new_batch_np.shape)

    # 调用 evaluate_grasp；返回 (B, 3, measure_dim)
    _, measures = evaluate_grasp(
        new_batch_np,          # already (2K,3,10)
        method="qd",
        device=device          # evaluate_grasp 内部可能忽略这个参数
    )
    # print(new_gt_measures.shape)
    new_gt_measures = measures.reshape(2 * K, 3, 2)
    # 取第 0 / 1 / 2 列作为 ref/p/n 的 measure
    new_ref_gt = new_gt_measures[:, 0]
    new_p_gt   = new_gt_measures[:, 1]
    new_n_gt   = new_gt_measures[:, 2]

    # ---------- 4. 生成干净标签 (+1/-1) ----------
    # new_gt_dis = (new_ref_gt - new_p_gt) - (new_ref_gt - new_n_gt)  # (2K,)
    new_gt_dis = np.sum(
        (np.square(new_ref_gt - new_p_gt) - np.square(new_ref_gt - new_n_gt)),
        axis=-1
    ) 
    
    # print(new_gt_dis.shape)
    new_lbl_clean = torch.from_numpy((new_gt_dis > 0).astype(np.float32))*2 - 1  # (+1,-1)
    new_lbl_clean = new_lbl_clean.to(device)

    # ---------- 5. 加噪（可选） ----------
    noise_gen = NoiseGenerator()
    new_lbl_noise = noise_gen.generate_noise(
        new_lbl_clean,
        new_gt_dis,
        noisy_method=noisy_method,
        parameter=parameter
    ).to(device)        # shape: (2K,)
    # print(new_lbl_noise.shape)
    # ---------- 6. 前向推断 delta_dis（不写入图） ----------
    with torch.no_grad():
        r_all   = new_batch[:, :10]      # (2K,10)
        p_all   = new_batch[:, 10:20]
        n_all   = new_batch[:, 20:30]

        delta_all = model.triplet_delta_dis(r_all, p_all, n_all)       # (2K,)
        # print(delta_all.shape)
        signed    = new_lbl_noise * delta_all                       # (2K,)

        best_idx  = torch.argmax(signed)

    # ---------- 7. 返回替换结果 ----------
    if best_idx < K:
        # 选择了 part1 → 更新 p
        new_p = p_expand[best_idx]
        new_n = batch2_i
        new_y = new_lbl_noise[best_idx]
        new_gt = new_lbl_clean[best_idx]
    else:
        # 选择了 part2 → 更新 n
        idx = best_idx - K
        new_p = batch1_i
        new_n = n_expand[idx]
        new_y = new_lbl_noise[best_idx]
        new_gt = new_lbl_clean[best_idx]

    return new_p, new_n, new_y, new_gt

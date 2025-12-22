import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads
class projection_head(nn.Module):
    def __init__(self, in_channels, input_height=1, input_width=1, hidden_dim=512, output_dim=1024):
        super().__init__()
        self.in_features = in_channels * input_height * input_width

        # 三层 MLP：in -> hidden -> hidden -> output
        self.mlp = nn.Sequential(
            nn.Flatten(),  # 将 (B, C, H, W) -> (B, C*H*W)
            nn.Linear(self.in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, output_dim)
            # 注意：最后一层通常不加 BN 和 ReLU（保留原始方向信息）
        )

    def forward(self, x):
        return self.mlp(x)


def loss_contrastive(z_r, z_i, y, temperature=0.2):
    """
    Contrastive loss for disentangled representation learning (strictly follows the given formula).

    Args:
        z_r: (N, D) task-relevant features
        z_i: (N, D) task-irrelevant features
        y:   (N,) class labels (integers >= 0, no background assumed)
        temperature: float > 0

    Returns:
        Scalar loss (mean over all valid anchors)
    """
    N, D = z_r.shape
    device = z_r.device

    if N < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Step 1: L2 normalize all features
    z_r = F.normalize(z_r, dim=1)
    z_i = F.normalize(z_i, dim=1)

    # Step 2: Build full feature bank Z_a = Z_r ∪ Z_i → (2N, D)
    Z_all = torch.cat([z_r, z_i], dim=0)  # (2N, D)

    # Step 3: Compute label equality mask (for positive pairs)
    y_eq = (y.unsqueeze(0) == y.unsqueeze(1))  # (N, N)
    eye = torch.eye(N, dtype=torch.bool, device=device)
    pos_mask = y_eq & (~eye)  # (N, N), exclude self

    total_loss = 0.0
    valid_count = 0

    # ---- Loss for z_r branch ----
    # Similarity between each z_r[j] and all features in Z_all
    sim_r_to_all = torch.mm(z_r, Z_all.t()) / temperature  # (N, 2N)
    log_prob_r = F.log_softmax(sim_r_to_all, dim=1)  # (N, 2N)

    # Positive indices: only first N columns (Z_r part), and only where pos_mask is True
    pos_logprob_r = log_prob_r[:, :N]  # (N, N)
    num_pos_r = pos_mask.sum(dim=1)  # (N,)

    for j in range(N):
        if num_pos_r[j] > 0:
            # Sum log-prob over positive samples for anchor j
            loss_j = -pos_logprob_r[j][pos_mask[j]].mean()
            total_loss += loss_j
            valid_count += 1

    # ---- Loss for z_i branch ----
    sim_i_to_all = torch.mm(z_i, Z_all.t()) / temperature  # (N, 2N)
    log_prob_i = F.log_softmax(sim_i_to_all, dim=1)  # (N, 2N)

    # pos_logprob_i = log_prob_i[:, :N]  # Wait! No!

    # ❗ Correction: For z_i, positive samples are in Z_i (second half)!
    # But note: the positive pairs are defined by same y, so we still use pos_mask on Z_i part
    pos_logprob_i = log_prob_i[:, N:]  # (N, N), corresponds to z_i vs z_i

    num_pos_i = pos_mask.sum(dim=1)  # same mask (same y condition)

    for j in range(N):
        if num_pos_i[j] > 0:
            loss_j = -pos_logprob_i[j][pos_mask[j]].mean()
            total_loss += loss_j
            valid_count += 1

    if valid_count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    output = total_loss / valid_count

    return output



def compute_entropy(s):
    """
    Compute entropy H(s) = -sum_c s(c) * log(s(c))
    Args:
        s: (N, C) probability distribution (should be normalized)
    Returns:
        entropy: (N,)
    """
    # 防止 log(0)
    s = torch.clamp(s, min=1e-8, max=1.0)
    return -(s * torch.log(s)).sum(dim=1)

def loss_entropy(z_tr, z_ti, pred_cls, roi_heads):
    #step1: 计算原始的熵
    pred_cls = F.softmax(pred_cls, dim=1)
    H_s = compute_entropy(pred_cls)  # (N,)

    #预测z_r和z_i的分类结果，使用RCNN中的分类头
    # z_tr_feature = roi_heads.box_head(z_tr)
    # z_ti_feature = roi_heads.box_head(z_ti)
    z_tr_predictions, _ = roi_heads.box_predictor(z_tr)
    z_ti_predictions, _ = roi_heads.box_predictor(z_ti)
    z_tr_predictions = F.softmax(z_tr_predictions, dim=1)
    z_ti_predictions = F.softmax(z_ti_predictions, dim=1)

    #Step2: 计算+和-的损失
    H_s_plus = compute_entropy(z_tr_predictions)  # (N,)
    H_s_minus = compute_entropy(z_ti_predictions)  # (N,)

    # Step 3: Compute L^+ and L^-
    L_plus = F.softplus(H_s_plus - H_s)  # encourage H(s_plus) < H(s)
    L_minus = F.softplus(H_s - H_s_minus)  # encourage H(s) < H(s_minus)

    # Step 4: Total loss (mean over batch)
    L_eco = (L_plus + L_minus).mean()

    return L_eco

def disentangle_loss(pooler_feature_r, pooler_feature_i, gt_cls, pred_cls, roi_heads):
    B, C, H, W = pooler_feature_r.shape
    projH = projection_head(in_channels=C, input_height=H, input_width=W).to('cuda')
    proj_tr, proj_ti = projH, projH
    z_tr = proj_tr(pooler_feature_r)   #(B,....)
    z_ti = proj_ti(pooler_feature_i)   #(B,....)
    contrastive_loss = loss_contrastive(z_tr, z_ti, gt_cls)
    entropy_loss = loss_entropy(z_tr, z_ti, pred_cls, roi_heads)

    return contrastive_loss, entropy_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) / Feed-Forward Network (FFN)

    Args:
        input_dim (int): 输入维度
        hidden_dim (int): 隐藏层维度（所有中间层都用这个）
        output_dim (int): 输出维度
        num_layers (int): 总层数（至少1层）

    Example:
        mlp = MLP(4, 128, 256, 3)  # 4 → 128 → 128 → 256
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        if num_layers == 1:
            self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        else:
            layers = [nn.Linear(input_dim, hidden_dim)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:  # 最后一层不加激活
                x = F.relu(x)
        return x

class token_encoder(nn.Module):
    def __init__(self, hidden_dim: int, box_mlp_layers: int = 3, score_mlp_layers: int = 2):
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim must be even for split encoding"
        self.d_half = hidden_dim // 2

        # Box encoder: (4) -> (d/2)
        self.box_embed = MLP(4, self.d_half, self.d_half, num_layers=box_mlp_layers)

        # Score encoder: (1) -> (d/2)
        self.score_embed = MLP(1, self.d_half, self.d_half, num_layers=score_mlp_layers)

    def forward(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:  # (N, M, d)
        N, M = boxes.shape[:2]

        # Encode boxes
        boxes_flat = boxes.view(-1, 4)  # (N*M, 4)
        box_tokens = self.box_embed(boxes_flat)  # (N*M, d/2)
        box_tokens = box_tokens.view(N, M, self.d_half)  # (N, M, d/2)

        # Encode scores
        scores_flat = scores.view(-1, 1)  # (N*M, 1)
        score_tokens = self.score_embed(scores_flat)  # (N*M, d/2)
        score_tokens = score_tokens.view(N, M, self.d_half)  # (N, M, d/2)

        # Concatenate
        tokens = torch.cat([box_tokens, score_tokens], dim=-1)  # (N, M, d)
        return tokens


class RewarderForProposals(nn.Module):
    """Adapted from original Rewarder to handle (B, M, D) inputs"""

    def __init__(self, input_dim=256):
        super().__init__()
        # 假设 gt_encoding 和 dt_encoding 都是 (B, M, D)
        # 我们将它们视为 N = B*M 个独立样本

        # Cross-Attention Mechanism (same as original)
        self.cross_attention_fc = nn.Linear(input_dim, 1)

        # MLP
        self.mlp_fc1 = nn.Linear(input_dim, 256)
        self.mlp_fc2 = nn.Linear(256, input_dim)

        # FFN
        self.ffn_fc1 = nn.Linear(input_dim, 64)
        self.ffn_fc2 = nn.Linear(64, 1)

        # LayerNorm (optional, but original has norm — you can add if needed)
        self.mlp_ln = nn.LayerNorm(input_dim)  # ✅ 这里必须是 input_dim（256）！
        self.ffn_ln = nn.LayerNorm(64)
        # Since your encodings are already processed, we skip extra norm here

    def forward(self, gt_encoding, dt_encoding):
        """
        Args:
            gt_encoding: (B, M, D)  ← treated as "label"
            dt_encoding: (B, M, D)  ← treated as "feature"
        Returns:
            reward: (B, M) in [0, 1]
        """
        B, M, D = gt_encoding.shape
        N = B * M

        # Reshape to (N, D)
        gt_flat = gt_encoding.view(N, D)  # (N, D)
        dt_flat = dt_encoding.view(N, D)  # (N, D)

        # Local cross-attention per proposal
        cross_attn_input = torch.stack([gt_flat, dt_flat], dim=1)  # (N, 2, D)
        logits = self.cross_attention_fc(cross_attn_input).squeeze(-1)  # (N, 2)
        weights = F.softmax(logits, dim=1)  # (N, 2)
        fused = (weights.unsqueeze(-1) * cross_attn_input).sum(dim=1)  # (N, D)

        # MLP Part
        mlp_out = F.relu(self.mlp_fc1(fused))  # (N, 256)
        mlp_out = self.mlp_fc2(mlp_out)  # (N, D)
        mlp_out = self.mlp_ln(mlp_out)  # ✅ LayerNorm on last dim (D=256)

        # FFN Part
        ffn_out = F.relu(self.ffn_fc1(mlp_out))  # (N, 64)
        ffn_out = self.ffn_ln(ffn_out)  # ✅ LayerNorm on 64
        logits = self.ffn_fc2(ffn_out)  # (N, 1)

        reward = torch.sigmoid(logits).view(B, M)  # (B, M)
        return reward



def compute_rewarder_loss(boxes_for_gt, scores_for_gt, gt_boxes_list, gt_classes_list, rewarder):
    boxes_for_gt_tensor = torch.stack(boxes_for_gt, dim=0) #tensor(4,512,4)
    scores_for_gt_tensor = torch.stack(scores_for_gt, dim=0) #tensor(4,512,)
    gt_boxes_tensor = torch.stack(gt_boxes_list, dim=0) #tensor(4,512,4)
    #gt_classes_tensor = torch.stack(gt_classes_list, dim=0) #tensor(4,512,)
    gt_classes_tensor = torch.ones_like(scores_for_gt_tensor)

    encoder = token_encoder(hidden_dim = 256).to('cuda')
    dt_encoding = encoder(boxes_for_gt_tensor, scores_for_gt_tensor) #tensor(4,512,256)
    with torch.no_grad():
        gt_encoding = encoder(gt_boxes_tensor, gt_classes_tensor)  # (4, 512, 256)

    r = rewarder(gt_encoding, dt_encoding) #r是奖励分数，越大说明dt和gt越接近
    loss_reward = (1.0-r).mean() #计算损失时应该取负号
    t = 0.5
    counts = (r > t).sum(dim=1)
    return loss_reward
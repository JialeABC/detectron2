import torch
import torch.nn as nn

# state_dict = torch.load("C:/Users/Administrator/Desktop/experiment/data2/model_0004999.pth")
# total_params = sum(param.numel() for param in state_dict.values())
# print(f"参数量: {total_params:,}")

checkpoint = torch.load("C:/Users/Administrator/Desktop/experiment/data2/model_0004999.pth", map_location="cpu")

# 判断类型
if isinstance(checkpoint, nn.Module):
    # 情况1: 保存的是整个模型
    model = checkpoint
    total_params = sum(p.numel() for p in model.parameters())
elif isinstance(checkpoint, dict):
    # 情况2: 保存的是 state_dict（或包含 state_dict 的 dict）
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint  # 直接是 state_dict

    # 确保所有值都是 Tensor
    total_params = 0
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            total_params += v.numel()
        else:
            print(f"⚠️ 跳过非张量项: {k} (type: {type(v)})")
else:
    raise TypeError(f"未知的 .pth 类型: {type(checkpoint)}")

print(f"总参数量: {total_params:,}")
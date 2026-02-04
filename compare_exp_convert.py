# import torch
# import pickle
#
# # ============================
# # 配置路径
# # ============================
# SW_PTH = "D:/Deeplearning_code/yolov8/detectron2/weight/resnet50_sw.pth"
# D2_PKL = "D:/Deeplearning_code/yolov8/detectron2/weight/model_final_280758.pkl"
# OUTPUT_PKL = "D:/Deeplearning_code/yolov8/detectron2/weight/model_final_with_sw_backbone.pkl"
#
# # ============================
# # 加载原始 Detectron2 模型
# # ============================
# print("📥 加载 model_final.pkl ...")
# with open(D2_PKL, "rb") as f:
#     d2_ckpt = pickle.load(f)
# original_model = d2_ckpt["model"]
#
# # ============================
# # 加载 SW 模型
# # ============================
# print("📥 加载 resnet50_sw.pth ...")
# sw_ckpt = torch.load(SW_PTH, map_location="cpu")
# if "state_dict" in sw_ckpt:
#     sw_state = sw_ckpt["state_dict"]
# elif "model" in sw_ckpt:
#     sw_state = sw_ckpt["model"]
# else:
#     sw_state = sw_ckpt
# if list(sw_state.keys())[0].startswith("module."):
#     sw_state = {k.replace("module.", ""): v for k, v in sw_state.items()}
#
#
# # ============================
# # Key 映射函数
# # ============================
# def map_key(k):
#     # Stem
#     if k == "conv1.weight":
#         return "backbone.bottom_up.stem.conv1.weight"
#     elif k == "sw1.weight":
#         return "backbone.bottom_up.stem.conv1.norm.weight"
#     elif k == "sw1.bias":
#         return "backbone.bottom_up.stem.conv1.norm.bias"
#
#     # 主体 layers: layer1→res2, layer2→res3, etc.
#     if k.startswith("layer"):
#         # 替换 layerX → res(X+1)
#         if k.startswith("layer1"):
#             k = k.replace("layer1", "res2", 1)
#         elif k.startswith("layer2"):
#             k = k.replace("layer2", "res3", 1)
#         elif k.startswith("layer3"):
#             k = k.replace("layer3", "res4", 1)
#         elif k.startswith("layer4"):
#             k = k.replace("layer4", "res5", 1)
#         else:
#             return None
#
#         # downsample → shortcut
#         k = k.replace("downsample.0", "shortcut")
#         k = k.replace("downsample.1", "shortcut.norm")
#
#         # bnX → convX.norm （注意：conv1/2/3 保持不变）
#         # 例如: res2.0.bn1.weight → res2.0.conv1.norm.weight
#         parts = k.split(".")
#         if len(parts) >= 4 and parts[2].startswith("bn"):
#             try:
#                 bn_idx = int(parts[2][2:])  # bn1 → 1
#                 conv_name = f"conv{bn_idx}"
#                 new_k = ".".join(parts[:2] + [conv_name, "norm"] + parts[3:])
#                 return f"backbone.bottom_up.{new_k}"
#             except:
#                 return None
#
#         # 其他情况（如 conv weight）直接加前缀
#         return f"backbone.bottom_up.{k}"
#
#     return None
#
#
# # ============================
# # 执行替换
# # ============================
# new_model = original_model.copy()
# replaced = 0
#
# for sw_k, tensor in sw_state.items():
#     # 跳过所有统计量和非可学习参数
#     if any(x in sw_k for x in [
#         "running_mean", "running_var", "num_batches_tracked",
#         "running_cov", "sw_mean_weight", "sw_var_weight"
#     ]):
#         continue
#
#     d2_k = map_key(sw_k)
#     if d2_k is None:
#         continue
#
#     if d2_k not in new_model:
#         print(f"⚠️ Key not in Detectron2: {d2_k}")
#         continue
#
#     if new_model[d2_k].shape != tensor.shape:
#         print(f"❌ Shape mismatch: {d2_k} | {new_model[d2_k].shape} vs {tensor.shape}")
#         continue
#
#     new_model[d2_k] = tensor
#     replaced += 1
#
# # ============================
# # 保存为 .pkl
# # ============================
# print(f"\n✅ 成功替换 {replaced} 个参数。")
#
# new_ckpt = {"model": new_model}
# for k, v in d2_ckpt.items():
#     if k != "model":
#         new_ckpt[k] = v
#
# with open(OUTPUT_PKL, "wb") as f:
#     pickle.dump(new_ckpt, f)
#
# print(f"🎉 已保存到: {OUTPUT_PKL}")
#=========================================================以上是SW的代码=======================================================================#

# import torch
# import pickle
#
# # ===========================================
# # 🔧 配置路径
# # ===========================================
# PKL_PATH = "D:/Deeplearning_code/yolov8/detectron2/weight/model_final_280758.pkl"  # Detectron2 原始模型
# PTH_PATH = "D:/Deeplearning_code/yolov8/detectron2/weight/CDSD/fpn_1_10_19317.pth"  # 你的 .pth 模型
# OUTPUT_PKL = "D:/Deeplearning_code/yolov8/detectron2/weight/CDSD/model_final_with_cdsd_backbone_rpn.pkl"
#
# # ===========================================
# # 加载模型
# # ===========================================
# # ====== 构建映射表 ======
# def build_resnet50_mapping():
#     mapping = {}
#     mapping["RCNN_layer0.0.weight"] = "backbone.bottom_up.stem.conv1.weight"
#     mapping["RCNN_layer0.1.weight"] = "backbone.bottom_up.stem.conv1.norm.weight"
#     mapping["RCNN_layer0.1.bias"] = "backbone.bottom_up.stem.conv1.norm.bias"
#
#     stages = [("RCNN_layer1", "res2", 3), ("RCNN_layer2", "res3", 4),
#               ("RCNN_layer3", "res4", 6), ("RCNN_layer4", "res5", 3)]
#
#     for pth_pre, d2_stage, n_blk in stages:
#         for bid in range(n_blk):
#             d2 = f"backbone.bottom_up.{d2_stage}.{bid}"
#             pth = f"{pth_pre}.{bid}.0"
#
#             # 3 convs
#             for i in [1, 2, 3]:
#                 mapping[f"{pth}.conv{i}.weight"] = f"{d2}.conv{i}.weight"
#                 mapping[f"{pth}.bn{i}.weight"] = f"{d2}.conv{i}.norm.weight"
#                 mapping[f"{pth}.bn{i}.bias"] = f"{d2}.conv{i}.norm.bias"
#
#             # shortcut (first block only)
#             if bid == 0:
#                 mapping[f"{pth}.downsample.0.weight"] = f"{d2}.shortcut.weight"
#                 mapping[f"{pth}.downsample.1.weight"] = f"{d2}.shortcut.norm.weight"
#                 mapping[f"{pth}.downsample.1.bias"] = f"{d2}.shortcut.norm.bias"
#     return mapping
#
#
# # ====== 加载模型 ======
# print("📥 加载 Detectron2 .pkl...")
# with open(PKL_PATH, "rb") as f:
#     d2_ckpt = pickle.load(f)
# d2_model = d2_ckpt["model"]
#
# print("📥 加载 .pth...")
# pth = torch.load(PTH_PATH, map_location="cpu")
# state = pth.get("state_dict", pth)
# if list(state.keys())[0].startswith("module."):
#     state = {k.replace("module.", ""): v for k, v in state.items()}
#
# # ====== 执行迁移 ======
# mapping = build_resnet50_mapping()
# new_model = d2_model.copy()
# replaced = 0
#
# for pth_k, tensor in state.items():
#     # 跳过 BN 统计量和非 backbone 参数
#     if any(x in pth_k for x in ["running_mean", "running_var", "num_batches_tracked", "RPN_", "roi"]):
#         continue
#
#     if pth_k not in mapping:
#         continue  # 不是我们要的 backbone 参数
#
#     d2_k = mapping[pth_k]
#     if d2_k not in new_model:
#         print(f"⚠️ Key not in .pkl: {d2_k}")
#         continue
#
#     if new_model[d2_k].shape != tensor.shape:
#         print(f"❌ Shape mismatch: {d2_k} | {new_model[d2_k].shape} vs {tensor.shape}")
#         continue
#
#     new_model[d2_k] = tensor
#     replaced += 1
#
# # ====== 保存 ======
# print(f"\n✅ 成功替换 {replaced} 个 backbone 参数")
#
# output_ckpt = {"model": new_model}
# for k, v in d2_ckpt.items():
#     if k != "model":
#         output_ckpt[k] = v
#
# with open(OUTPUT_PKL, "wb") as f:
#     pickle.dump(output_ckpt, f)
#
# print(f"🎉 保存成功: {OUTPUT_PKL}")
# print("\n📌 RPN 和 ROI Head 保持原样，将在训练中微调。")
#========================================================上面是CSDS代码===================================================================#

# import torch
# import pickle
#
# # ====== 路径配置 ======
# PKL_PATH = "D:/Deeplearning_code/yolov8/detectron2/weight/model_final_280758.pkl"
# PTH_PATH = "D:/Deeplearning_code/yolov8/detectron2/weight/ISW/last_cityscapes_epoch_108_mean-iu_0.76084.pth"
# OUTPUT_PKL = "D:/Deeplearning_code/yolov8/detectron2/weight/ISW/model_with_isw_backbone.pkl"
#
#
# # ====== 构建映射 ======
# def build_resnet50_mapping_for_isw():
#     mapping = {}
#     mapping["module.layer0.0.weight"] = "backbone.bottom_up.stem.conv1.weight"
#     mapping["module.layer0.1.weight"] = "backbone.bottom_up.stem.conv1.norm.weight"
#     mapping["module.layer0.1.bias"] = "backbone.bottom_up.stem.conv1.norm.bias"
#
#     def add_block(pth_pre, d2_pre, is_first):
#         for i in [1, 2, 3]:
#             mapping[f"{pth_pre}.conv{i}.weight"] = f"{d2_pre}.conv{i}.weight"
#             mapping[f"{pth_pre}.bn{i}.weight"] = f"{d2_pre}.conv{i}.norm.weight"
#             mapping[f"{pth_pre}.bn{i}.bias"] = f"{d2_pre}.conv{i}.norm.bias"
#         if is_first:
#             mapping[f"{pth_pre}.downsample.0.weight"] = f"{d2_pre}.shortcut.weight"
#             mapping[f"{pth_pre}.downsample.1.weight"] = f"{d2_pre}.shortcut.norm.weight"
#             mapping[f"{pth_pre}.downsample.1.bias"] = f"{d2_pre}.shortcut.norm.bias"
#
#     # Res2
#     for i in range(3):
#         add_block(f"module.layer1.{i}", f"backbone.bottom_up.res2.{i}", i == 0)
#     # Res3
#     for i in range(4):
#         add_block(f"module.layer2.{i}", f"backbone.bottom_up.res3.{i}", i == 0)
#     # Res4
#     for i in range(6):
#         add_block(f"module.layer3.{i}", f"backbone.bottom_up.res4.{i}", i == 0)
#     # Res5
#     for i in range(3):
#         add_block(f"module.layer4.{i}", f"backbone.bottom_up.res5.{i}", i == 0)
#
#     return mapping
#
#
# # ====== 加载模型 ======
# print("📥 加载 Detectron2 .pkl...")
# with open(PKL_PATH, "rb") as f:
#     d2_ckpt = pickle.load(f)
# d2_model = d2_ckpt["model"]
#
# print("📥 加载 ISW .pth...")
# pth = torch.load(PTH_PATH, map_location="cpu")
# state = pth["state_dict"]  # 从日志看，权重在 'state_dict'
#
# # ====== 执行迁移 ======
# mapping = build_resnet50_mapping_for_isw()
# new_model = d2_model.copy()
# replaced = 0
#
# for pth_k, tensor in state.items():
#     if pth_k not in mapping:
#         continue
#
#     d2_k = mapping[pth_k]
#     if d2_k not in new_model:
#         print(f"⚠️ Key not in .pkl: {d2_k}")
#         continue
#
#     if new_model[d2_k].shape != tensor.shape:
#         print(f"❌ Shape mismatch: {d2_k} | {new_model[d2_k].shape} vs {tensor.shape}")
#         continue
#
#     new_model[d2_k] = tensor
#     replaced += 1
#
# # ====== 保存 ======
# print(f"\n✅ 成功替换 {replaced} 个 backbone 参数")
#
# output_ckpt = {"model": new_model}
# for k, v in d2_ckpt.items():
#     if k != "model":
#         output_ckpt[k] = v
#
# with open(OUTPUT_PKL, "wb") as f:
#     pickle.dump(output_ckpt, f)
#
# print(f"🎉 保存成功: {OUTPUT_PKL}")
# print("\n📌 FPN、RPN、ROI Head 保持原样，将在训练中微调。")
#========================================================以上是ISW的代码=================================================================#

import torch
import pickle

PKL_PATH = "D:/Deeplearning_code/yolov8/detectron2/weight/model_final_280758.pkl"
PTH_PATH = "D:/Deeplearning_code/yolov8/detectron2/weight/IBN-Net/resnet50_ibn_a-d9d0bb7b.pth"
OUTPUT_PKL = "D:/Deeplearning_code/yolov8/detectron2/weight/IBN-Net/model_with_ibn_backbone_no_bn1.pkl"


def build_ibn_resnet50_to_detectron2_mapping():
    mapping = {}
    # Stem is full BN
    mapping["conv1.weight"] = "backbone.bottom_up.stem.conv1.weight"
    mapping["bn1.weight"] = "backbone.bottom_up.stem.conv1.norm.weight"
    mapping["bn1.bias"] = "backbone.bottom_up.stem.conv1.norm.bias"

    def add_block(stage, block_idx):
        pth = f"layer{stage}.{block_idx}"
        d2 = f"backbone.bottom_up.res{stage + 1}.{block_idx}"

        # Only transfer conv weights and bn2/bn3
        mapping[f"{pth}.conv1.weight"] = f"{d2}.conv1.weight"
        # SKIP bn1 (conv1.norm) due to IBN half-channel issue

        mapping[f"{pth}.conv2.weight"] = f"{d2}.conv2.weight"
        mapping[f"{pth}.bn2.weight"] = f"{d2}.conv2.norm.weight"
        mapping[f"{pth}.bn2.bias"] = f"{d2}.conv2.norm.bias"

        mapping[f"{pth}.conv3.weight"] = f"{d2}.conv3.weight"
        mapping[f"{pth}.bn3.weight"] = f"{d2}.conv3.norm.weight"
        mapping[f"{pth}.bn3.bias"] = f"{d2}.conv3.norm.bias"

        if block_idx == 0:
            mapping[f"{pth}.downsample.0.weight"] = f"{d2}.shortcut.weight"
            mapping[f"{pth}.downsample.1.weight"] = f"{d2}.shortcut.norm.weight"
            mapping[f"{pth}.downsample.1.bias"] = f"{d2}.shortcut.norm.bias"

    for i in range(3): add_block(1, i)
    for i in range(4): add_block(2, i)
    for i in range(6): add_block(3, i)
    for i in range(3): add_block(4, i)

    return mapping


# Load
print("📥 加载 Detectron2 .pkl...")
with open(PKL_PATH, "rb") as f:
    d2_ckpt = pickle.load(f)
d2_model = d2_ckpt["model"]

print("📥 加载 IBN-Net .pth...")
state = torch.load(PTH_PATH, map_location="cpu", weights_only=True)

# Transfer
mapping = build_ibn_resnet50_to_detectron2_mapping()
new_model = d2_model.copy()
replaced = 0

for pth_k, tensor in state.items():
    if pth_k not in mapping:
        continue
    d2_k = mapping[pth_k]
    if d2_k not in new_model:
        print(f"⚠️ Key not in .pkl: {d2_k}")
        continue
    if new_model[d2_k].shape != tensor.shape:
        print(f"❌ Shape mismatch: {d2_k} | {new_model[d2_k].shape} vs {tensor.shape}")
        continue
    new_model[d2_k] = tensor
    replaced += 1

# Save
print(f"\n✅ 成功替换 {replaced} 个 backbone 参数（不含 bn1）")
output_ckpt = {"model": new_model}
for k, v in d2_ckpt.items():
    if k != "model":
        output_ckpt[k] = v

with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(output_ckpt, f)

print(f"🎉 保存成功: {OUTPUT_PKL}")
print("📌 提示：conv1.norm (bn1) 将保持原 .pkl 中的值（或随机初始化），将在训练中微调。")
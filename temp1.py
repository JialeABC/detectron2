import torch

# 加载 .pth 文件
checkpoint = torch.load('D:/Deeplearning_code/yolov8/detectron2/tools/output/model_0011999.pth', map_location='cpu')  # 推荐用 cpu 避免 GPU 冲突

# 查看顶层结构类型
print(type(checkpoint))

# 如果是 dict，打印所有 key
if isinstance(checkpoint, dict):
    print("Keys in the checkpoint:")
    for key in checkpoint.keys():
        print(f"  - {key}")

    res = []
    # 可选：查看每个 key 对应的值的类型和形状（如果是 tensor）
    for key, value in checkpoint.items():
        if key=="model":
            for i in value:
                res.append(i)
        else:
            break
    for item in res:
        print(item)


# import torch
# checkpoint = torch.load("/home/xwwen/EAGLE_test/training_data/sharegpt_0_67_mufp16/0/data_0.ckpt", map_location="cpu")
# print(checkpoint.keys())  # 查看有哪些内容

import torch

# 加载 checkpoint
checkpoint_path = "/home/xwwen/EAGLE_test/training_data/sharegpt_0_67_mufp16/0/data_0.ckpt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# 打印所有键
print("Checkpoint 包含的键：")
for key in checkpoint.keys():
    value = checkpoint[key]
    if isinstance(value, torch.Tensor):
        print(f"{key}: Tensor, shape={value.shape}, dtype={value.dtype}")
    elif isinstance(value, dict):
        print(f"{key}: Dict, keys={list(value.keys())}")
    elif isinstance(value, list):
        print(f"{key}: List, length={len(value)}")
    elif isinstance(value, str):
        print(f"{key}: String, content (truncated)={value[:50]}")
    elif isinstance(value, int) or isinstance(value, float):
        print(f"{key}: {type(value).__name__}, value={value}")
    else:
        print(f"{key}: {type(value).__name__}")

# 如果需要查看具体内容，可根据键选择打印
example_key = next(iter(checkpoint.keys()))  # 选择一个键
print(f"\n示例 Key ({example_key}) 的内容预览:")
print(checkpoint[example_key])


# from safetensors import safe_open

# file_path = "/home/xwwen/EAGLE_test/Llama-2-7b-chat-hf/model-00001-of-00002.safetensors"

# with safe_open(file_path, framework="pt", device="cpu") as f:
#     tensors = f.keys()  # 获取所有张量的键名
#     for tensor_name in tensors:
#         tensor_slice = f.get_slice(tensor_name)  # 获取张量的切片对象
#         shape = tensor_slice.get_shape()  # 获取张量的形状
#         print(f"Tensor: {tensor_name}, Shape: {shape}")

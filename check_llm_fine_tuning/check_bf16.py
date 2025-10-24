import torch

print(torch.cuda.is_available())
print(torch.cuda.is_tf32_supported())
print(torch.cuda.is_bf16_supported(True))
print(torch.cuda.is_bf16_supported(False))
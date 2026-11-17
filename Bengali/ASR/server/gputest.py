import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))

# real compute test - this is what fails if the CUDA build doesn't match the GPU
x = torch.randn(4000, 4000, device="cuda")
y = (x @ x).sum().item()
print("GPU MATH OK:", y == y)          # False if NaN -> broken

free, total = torch.cuda.mem_get_info()
print(f"VRAM free: {free/1e9:.1f} GB / {total/1e9:.1f} GB")

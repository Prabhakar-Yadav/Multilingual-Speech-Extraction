import torch
x = torch.randn(500, 500, device="cuda")
ok = ((x @ x).sum().item() == (x @ x).sum().item())
print("GPU_OK", ok)
f, t = torch.cuda.mem_get_info()
print(f"FREE {f/1e9:.1f} GB of {t/1e9:.1f} GB")

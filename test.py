import torch

x = torch.randn((1024, 1024, 1024), device="cuda")  # Adjust size as needed
print("Tensor allocated successfully.")
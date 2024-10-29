import torch

p = torch.randn(12,28,28)
p1 = p[0].view(-1).unsqueeze(0)
print(p1.shape)
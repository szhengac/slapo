import torch
from apex.optimizers import FusedAdam

t = torch.zeros(2359332864, dtype=torch.float, device='cuda')
t.grad = torch.zeros_like(t)
params = [t]
optimizer = FusedAdam(params)
optimizer.step()
torch.cuda.synchronize()

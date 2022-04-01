# coding=utf-8

import torch
import dadt.pytorch as dadt

dadt.initialize(executor_type='nccl')

device = torch.device(f'cuda:{dadt.local_rank()}')

if 0 == dadt.rank():
  x = torch.tensor([1, 2, 3, 4], device=device)
else:
  x = torch.tensor([5, 4, 3, 2], device=device)

y = dadt.all_reduce(0, x)
print(dadt.rank(), y)

y = dadt.all_reduce(0, x)
print(dadt.rank(), y)

dadt.shutdown()

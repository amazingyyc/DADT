#coding=utf-8

import torch
import ctypes
import dadt.pytorch as dadt

dadt.init(broad_cast_executor='mpi', all_reduce_executor='nccl')

device = torch.device('cuda:{}'.format(dadt.local_rank()))
# device = torch.device('cpu')

if 0 == dadt.rank():
  x = torch.tensor([1, 2, 3, 4], device=device, dtype=torch.float)
else:
  x = torch.tensor([1, 1, 1, 1], device=device, dtype=torch.float)

y = dadt.all_reduce(x, "x")

print(dadt.rank(), y)

y = dadt.all_reduce(x, "x")

print(dadt.rank(), y)

dadt.shutdown()
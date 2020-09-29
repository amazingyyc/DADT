#coding=utf-8

import torch
import ctypes
import dadt.pytorch as dadt

dadt.init(broad_cast_executor='mpi')

device = torch.device('cuda:{}'.format(dadt.local_rank()))

if 0 == dadt.rank():
  x = torch.tensor([1, 2, 3, 4], device=device)
else:
  x = torch.tensor([0, 0, 0, 0], device=device)

y = dadt.broad_cast(x, "x")

print(dadt.rank(), y)

dadt.shutdown()
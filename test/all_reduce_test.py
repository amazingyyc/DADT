# coding=utf-8

import torch
import dadt.pytorch as dadt

dadt.initialize()

if 0 == dadt.rank():
  x = torch.tensor([1, 2, 3, 4])
else:
  x = torch.tensor([1, 2, 3, 5])

y = dadt.all_reduce_async(0, x)

print(dadt.rank(), y)

y = dadt.all_reduce_async(0, x)

print(dadt.rank(), y)

dadt.shutdown()

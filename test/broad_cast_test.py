# coding=utf-8

import torch
import dadt.pytorch as dadt

dadt.initialize()

if 0 == dadt.rank():
  x = torch.tensor([100, 20, 30, 400])
else:
  x = torch.tensor([5, 4, 3, 2])

y = dadt.broad_cast(0, x)

print(dadt.rank(), y)

y = dadt.broad_cast(0, x)

print(dadt.rank(), y)

dadt.shutdown()

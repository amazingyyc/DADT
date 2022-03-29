# coding=utf-8

import torch
import dadt.pytorch as dadt

config = dadt.Config()
config.cycle_duration_ms = 3
config.broad_cast_executor = 'mpi'
config.all_reduce_executor = 'mpi'
config.all_reduce_buffer_size = 64 * 1024 * 1024

dadt.initialize(config)

if 0 == dadt.rank():
  x = torch.tensor([1, 2, 3, 4])
else:
  x = torch.tensor([1, 2, 3, 5])

y = dadt.all_reduce_async(0, x)

print(dadt.rank(), y)

y = dadt.all_reduce_async(0, x)

print(dadt.rank(), y)

dadt.shutdown()

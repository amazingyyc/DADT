# coding=utf-8

import torch
import dadt.pytorch as dadt

config = dadt.Config()
config.cycle_duration_ms = 3
config.broad_cast_executor = 'mpi'
config.all_reduce_executor = 'mpi'
config.all_reduce_buffer_size = 64 * 1024 * 1024

dadt.initialize(config)

print('size:', dadt.size())
print('rank:', dadt.rank())

if 0 == dadt.rank():
  x = torch.tensor([1, 2, 3, 4])
else:
  x = torch.tensor([4, 3, 2, 1])

# x = dadt.broad_cast_cpu(0, x)
x = dadt.all_reduce_cpu(1, x)

print('Rank:', dadt.rank(), x)

dadt.shutdown()

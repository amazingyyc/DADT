# coding=utf-8

import torch
import dadt.pytorch as dadt

config = dadt.Config()
config.cycle_duration_ms = 3
config.broad_cast_executor = 'nccl'
config.all_reduce_executor = 'nccl'
config.all_reduce_buffer_size = 64 * 1024 * 1024
config.gpu_device_id = 0

device = torch.device('cuda:0')

dadt.initialize(config)

if 0 == dadt.rank():
  x = torch.tensor([1, 2, 3, 4], device=device)
else:
  x = torch.tensor([5, 4, 3, 2], device=device)

y = dadt.broad_cast(0, x)

print(dadt.rank(), y)

y = dadt.broad_cast(0, x)

print(dadt.rank(), y)

dadt.shutdown()

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
  i = [[0, 0], [0, 1]]
  v = [[3, 4], [5, 6]]

  x = torch.sparse_coo_tensor(i, v, (2, 3, 2))
else:
  i = [[1, 1, 1], [0, 1, 2]]
  v = [[3, 4], [5, 6], [7, 8]]

  x = torch.sparse_coo_tensor(i, v, (2, 3, 2))

x = x.coalesce()
x = dadt.coo_all_reduce_async(0, x)

print('rank:', dadt.rank(), x.to_dense())

x = x.coalesce()
x = dadt.coo_all_reduce_async(0, x)

print('rank:', dadt.rank(), x.to_dense())

dadt.shutdown()

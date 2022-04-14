# coding=utf-8

import torch
import dadt.pytorch as dadt

# coding=utf-8

import torch
import numpy as np
import dadt.pytorch as dadt


def test_all_reduce_gpu(id):
  device = torch.device(f'cuda:{dadt.local_rank()}')

  length = 100

  if dadt.rank() == 0:
    x = torch.tensor(range(length), device=device)
  else:
    x = torch.tensor(range(length, 0, -1), device=device)

  expect = torch.tensor(range(length), device=device) + torch.tensor(
      range(length, 0, -1), device=device)
  real = dadt.all_reduce(id, x)

  torch.testing.assert_close(expect, real)

  print('------------------------------------------')
  print('[TestAllReduceGpu Success!]')
  print('------------------------------------------')


if __name__ == '__main__':
  dadt.initialize(executor_type='nccl')
  test_all_reduce_gpu(0)
  dadt.shutdown()

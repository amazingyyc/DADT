# coding=utf-8

import torch
import numpy as np
import dadt.pytorch as dadt


def test_broad_cast_gpu(id):
  device = torch.device(f'cuda:{dadt.local_rank()}')

  length = 10

  expect = np.asarray(range(0, length, 1))
  expect = torch.from_numpy(expect).to(device)

  if dadt.rank() == 0:
    # 0 -> lenght-1
    x = expect
  else:
    # length-1 -> 0
    x = np.asarray(range(length, 0, -1))
    x = torch.from_numpy(x).to(device)

  real = dadt.broad_cast(id, x)

  torch.testing.assert_close(expect, real)

  print('------------------------------------------')
  print('[TestBroadCastGpu Success!]')
  print('------------------------------------------')


if __name__ == '__main__':
  dadt.initialize(executor_type='nccl')
  test_broad_cast_gpu(0)
  dadt.shutdown()

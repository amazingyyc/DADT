# coding=utf-8

import torch
import numpy as np
import dadt.pytorch as dadt


def test_broad_cast_cpu(id):
  length = 10

  expect = np.asarray(range(0, length, 1))
  expect = torch.from_numpy(expect)

  if dadt.rank() == 0:
    # 0 -> lenght-1
    x = expect
  else:
    # length-1 -> 0
    x = np.asarray(range(length, 0, -1))
    x = torch.from_numpy(x)

  real = dadt.broad_cast(id, x)

  torch.testing.assert_close(expect, real)

  print('------------------------------------------')
  print('[TestBroadCastCpu Success!]')
  print('------------------------------------------')


if __name__ == '__main__':
  dadt.initialize()
  test_broad_cast_cpu(0)
  dadt.shutdown()

# coding=utf-8

import torch
import dadt.pytorch as dadt

# coding=utf-8

import torch
import numpy as np
import dadt.pytorch as dadt


def test_all_reduce_cpu(id):
  length = 100

  if dadt.rank() == 0:
    x = torch.tensor(range(length))
  else:
    x = torch.tensor(range(length, 0, -1))

  expect = torch.tensor(range(length)) + torch.tensor(range(length, 0, -1))
  real = dadt.all_reduce(id, x)

  torch.testing.assert_close(expect, real)

  print('------------------------------------------')
  print('[TestAllReduceCpu Success!]')
  print('------------------------------------------')


if __name__ == '__main__':
  dadt.initialize()
  test_all_reduce_cpu(0)
  dadt.shutdown()

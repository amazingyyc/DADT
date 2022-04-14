# coding=utf-8

import torch
import dadt.pytorch as dadt

# coding=utf-8

import torch
import numpy as np
import dadt.pytorch as dadt


def test_all_reduce_async_cpu(id):
  length = 100

  if dadt.rank() == 0:
    x = torch.tensor(range(length))
  else:
    x = torch.tensor(range(length, 0, -1))

  expect = torch.tensor(range(length)) + torch.tensor(range(length, 0, -1))
  zeros = torch.zeros(length, dtype=expect.dtype)

  real = dadt.all_reduce_async(id, x)
  torch.testing.assert_close(zeros, real)

  real = dadt.all_reduce_async(id, x)
  torch.testing.assert_close(expect, real)

  print('------------------------------------------')
  print('[TestAllReduceAsyncCpu Success!]')
  print('------------------------------------------')


if __name__ == '__main__':
  dadt.initialize()
  test_all_reduce_async_cpu(0)
  dadt.shutdown()

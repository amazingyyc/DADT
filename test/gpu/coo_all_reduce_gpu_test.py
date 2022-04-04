# coding=utf-8

import torch
import dadt.pytorch as dadt


def test_coo_all_reduce_gpu(id):
  device = torch.device(f'cuda:{dadt.local_rank()}')

  if dadt.rank() == 0:
    i = [[0, 0, 1], [0, 1, 0]]
    v = [[3, 4], [5, 6], [10, 10]]

    x = torch.sparse_coo_tensor(i, v, (2, 3, 2), device=device)
  else:
    i = [[1, 1, 1], [0, 1, 2]]
    v = [[3, 4], [5, 6], [7, 8]]

    x = torch.sparse_coo_tensor(i, v, (2, 3, 2), device=device)

  i = [[0, 0, 1, 1, 1, 1], [0, 1, 0, 0, 1, 2]]
  v = [[3, 4], [5, 6], [10, 10], [3, 4], [5, 6], [7, 8]]
  expect = torch.sparse_coo_tensor(i, v, (2, 3, 2), device=device).coalesce()

  real = dadt.coo_all_reduce(0, x.coalesce())

  torch.testing.assert_close(expect, real)

  print('------------------------------------------')
  print('[TestCooAllReduceCpu Success!]')
  print('------------------------------------------')


if __name__ == '__main__':
  dadt.initialize(executor_type='nccl')
  test_coo_all_reduce_gpu(0)
  dadt.shutdown()

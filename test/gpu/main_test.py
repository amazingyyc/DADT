# coding=utf-8

import unittest
import dadt.pytorch as dadt
from broad_cast_gpu_test import test_broad_cast_gpu
from all_reduce_gpu_test import test_all_reduce_gpu
from all_reduce_async_gpu_test import test_all_reduce_async_gpu
from coo_all_reduce_gpu_test import test_coo_all_reduce_gpu


def run_test():
  test_broad_cast_gpu(id=0)
  test_all_reduce_gpu(id=1)
  test_all_reduce_async_gpu(id=2)
  test_coo_all_reduce_gpu(id=3)


if __name__ == '__main__':
  dadt.initialize(executor_type='nccl')
  run_test()
  dadt.shutdown()

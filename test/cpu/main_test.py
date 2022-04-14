# coding=utf-8

import unittest
import dadt.pytorch as dadt
from broad_cast_cpu_test import test_broad_cast_cpu
from all_reduce_cpu_test import test_all_reduce_cpu
from all_reduce_async_cpu_test import test_all_reduce_async_cpu
from coo_all_reduce_cpu_test import test_coo_all_reduce_cpu


def run_test():
  test_broad_cast_cpu(id=0)
  test_all_reduce_cpu(id=1)
  test_all_reduce_async_cpu(id=2)
  test_coo_all_reduce_cpu(id=3)


if __name__ == '__main__':
  dadt.initialize()
  run_test()
  dadt.shutdown()

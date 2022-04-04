# coding=utf-8

import dadt_pytorch

from dadt_pytorch import Config
from dadt_pytorch import shutdown
from dadt_pytorch import initialized
from dadt_pytorch import size
from dadt_pytorch import local_size
from dadt_pytorch import rank
from dadt_pytorch import local_rank
from dadt_pytorch import barrier
from dadt_pytorch import local_barrier
from dadt_pytorch import broad_cast
from dadt_pytorch import all_reduce
from dadt_pytorch import all_reduce_async
from dadt_pytorch import coo_all_reduce
from dadt_pytorch import coo_all_reduce_async
from .distributed_optimizer import DistributedOptimizer


def initialize(cycle_duration_ms=5,
               executor_type='mpi',
               all_reduce_buffer_size=64 * 1024 * 1024,
               gpu_device_id=-1):
  config = dadt_pytorch.Config()
  config.cycle_duration_ms = cycle_duration_ms
  config.executor_type = executor_type
  config.all_reduce_buffer_size = all_reduce_buffer_size
  config.gpu_device_id = gpu_device_id

  dadt_pytorch.initialize(config)

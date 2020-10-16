# coding=utf-8

import torch
from dadt.pytorch import dadt_pytorch as native_ops
from dadt import dadt_lib_module

'''broad the tensor from rank 0 to other rank'''
def broad_cast(x, name):
  return native_ops.broad_cast(x, name)

'''do allreduce for all rank'''
def all_reduce(x, name, multiplier=1.0):
  return native_ops.all_reduce(x, name, multiplier)

'''do allreduce async for all rank'''
# def all_reduce_async(x, y, name, multiplier=1.0):
#   native_ops.all_reduce_async(x, y, name, multiplier)

'''wait allreduce finish'''
# def wait_all_reduce_finish():
#   native_ops.wait_all_reduce_finish()

class DistributedOptimizer:
  # DistributedOptimizer idx every DistributedOptimizer has a unique index
  _DistributedIndex = 0

  @classmethod
  def _generate_index(cls):
    cls._DistributedIndex += 1
    return cls._DistributedIndex

  def __init__(self, optimizer: torch.optim.Optimizer, is_avg=True):
    self._optimizer = optimizer
    self._is_avg = is_avg
    self._parameter_names = {}

    # every DistributedOptimizer has a uniuqe index
    self._index = DistributedOptimizer._generate_index()

    if self._is_avg:
      self._multiplier = float(1.0 / float(dadt_lib_module.size()))
    else:
      self._multiplier = float(1.0)

    '''when initialize the DistributedOptimizer we need broadcast trainable parameter from rank 0 to other ranks'''
    broad_cast_idx = 0

    for param_group in self._optimizer.param_groups:
      for p in param_group['params']:
        if p.requires_grad:
          # use short string, b means broadcast
          name = '{}.b.{}'.format(self._index, broad_cast_idx)
          p.data = broad_cast(p.data, name)

          broad_cast_idx += 1

    '''at here add a hook for every trainable's grad to allreduce grad before run step()'''
    all_reduce_idx = 0

    for param_group in self._optimizer.param_groups:
      for p in param_group['params']:
        if p.requires_grad:
          # store unique name, use short string, a means allreduce
          self._parameter_names[p] = '{}.a.{}'.format(self._index, all_reduce_idx)
          all_reduce_idx += 1

          p.grad = p.data.new(p.size()).zero_()
          p.register_hook(self._create_grad_hook(p))

  def _create_grad_hook(self, p):
    def hook(grad):
      name = self._parameter_names[p]
      return all_reduce(grad, name, self._multiplier)
      # name = self._parameter_names[p]
      # new_grad = torch.zeros_like(grad)
      # all_reduce_async(grad, new_grad, name, self._multiplier)
      # return new_grad

    return hook

  def step(self, closure=None):
    # wait allreduce finish
    # wait_all_reduce_finish()

    # update gradient
    self._optimizer.step(closure=closure)

  def zero_grad(self):
    self._optimizer.zero_grad()

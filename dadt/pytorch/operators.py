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

class DistributedOptimizer:
  def __init__(self, optimizer: torch.optim.Optimizer, is_avg=True):
    self._optimizer = optimizer
    self._is_avg = is_avg
    self._parameter_names = {}

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
          name = 'b.{}'.format(broad_cast_idx)

          p.data = broad_cast(p.data, name)

          broad_cast_idx += 1

    '''at here add a hook for every trainable's grad to allreduce grad before run step()'''
    all_reduce_idx = 0

    for param_group in self._optimizer.param_groups:
      for p in param_group['params']:
        if p.requires_grad:
          # store unique name, use short string, a means allreduce
          self._parameter_names[p] = 'a.{}'.format(all_reduce_idx)
          all_reduce_idx += 1

          p.grad = p.data.new(p.size()).zero_()
          p.register_hook(self._create_grad_hook(p))

  def _create_grad_hook(self, p):
    def hook(grad):
      name = self._parameter_names[p]
      return all_reduce(grad, name, self._multiplier)

    return hook

  def step(self, closure=None):
    self._optimizer.step(closure=closure)

  def zero_grad(self):
    self._optimizer.zero_grad()

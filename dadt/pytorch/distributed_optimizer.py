# coding=utf-8

import torch
import dadt_pytorch


class DistributedOptimizer(object):

  def __init__(self, optimizer: torch.optim.Optimizer, avg: bool = True):
    assert dadt_pytorch.initialized() == True

    self._optimizer = optimizer
    self._avg = avg
    self._multiplier = 1.0

    if self._avg:
      self._multiplier = 1.0 / float(dadt_pytorch.size())

    self._param_ids = dict()

    # assign a unique id for every parameter
    _id = 0

    for param_group in self._optimizer.param_groups:
      for p in param_group['params']:
        if p.requires_grad:
          self._param_ids[p] = _id
          _id += 1

    # BroadCast the param to other rank
    for param_group in self._optimizer.param_groups:
      for p in param_group['params']:
        if p.requires_grad:
          p.data = dadt_pytorch.broad_cast(self._param_ids[p], p.data)

    # Create grad hook for param
    for param_group in self._optimizer.param_groups:
      for p in param_group['params']:
        if p.requires_grad:
          p.register_hook(self._create_grad_hook(self._param_ids[p]))

  def _create_grad_hook(self, id):

    def hook(grad):
      if grad.is_sparse:
        return dadt_pytorch.coo_all_reduce_async(id, grad)
      else:
        return dadt_pytorch.all_reduce_async(id, grad)

    return hook

  def step(self, closure=None):
    self._optimizer.step(closure=closure)

  def zero_grad(self):
    self._optimizer.zero_grad()

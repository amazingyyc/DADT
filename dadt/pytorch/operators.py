# coding=utf-8

from dadt.pytorch import dadt_pytorch as native_ops

'''broad the tensor from rank 0 to other rank'''
def broad_cast(x, name):
  return native_ops.broad_cast(x, name)

'''do allreduce for all rank'''
def all_reduce(x, name):
  return native_ops.all_reduce(x, name)

# coding=utf-8

import re
import ctypes
import platform
import tensorflow as tf

from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

if 'Windows' == platform.system():
  dadt_library_suffix = '.dll'
elif 'Linux' == platform.system():
  dadt_library_suffix = '.so'
elif 'Darwin' == platform.system():
  dadt_library_suffix = '.dylib'
else:
  raise ValueError('unsupported system')

'''load dadt library'''
dadt_library_path = resource_loader.get_path_to_datafile('../libdadt{0}'.format(dadt_library_suffix))

dadt_tf_module  = tf.load_op_library(dadt_library_path)
dadt_native_module = ctypes.CDLL(dadt_library_path, mode=ctypes.RTLD_GLOBAL)

'''init dadt'''
def init():
  dadt_native_module.init()

def shutdown():
  dadt_native_module.shutdown()

'''if have been initialized'''
def initialized():
  return dadt_native_module.initialized()

'''how many process'''
def size():
  return dadt_native_module.size()

'''how many process in current machine'''
def local_size():
  return dadt_native_module.local_size()

'''the rank of current process'''
def rank():
  return dadt_native_module.rank()

'''local rank'''
def local_rank():
  return dadt_native_module.local_rank()

'''barrier all process'''
def barrier():
  dadt_native_module.barrier()

'''local barrier all process'''
def local_barrier():
  dadt_native_module.local_barrier()

'''normalize the name'''
def normalize_name(name):
    return re.sub('[^a-zA-Z0-9_]', '_', name)

'''
all reduce op, the allreduce is async op, so when firt call this function it will return 0 
'''
def all_reduce(tensor, name=None):
  if name is None:
    name = 'DadtAllReduce_{0}'.format(normalize_name(tensor.name))

  return dadt_tf_module.dadt_all_reduce(tensor, name=name)

'''broad the tensor from rank 0 to other rank'''
def broad_cast(tensor, name=None):
  if name is None:
    name = 'DadtBroadCast_{0}'.format(normalize_name(tensor.name))
  
  return dadt_tf_module.dadt_broad_cast(tensor, name=name)

'''a session hook, will broad cast the weigths from rank 0 to other ranks'''
class BroadcastTrainableVariablesHook(tf.train.SessionRunHook):
  def __init__(self):
    super(BroadcastTrainableVariablesHook, self).__init__()

    self.broad_cast_op = None

  def begin(self):
    self.broad_cast_op = tf.group(*[tf.assign(var, broad_cast(var)) for var in tf.trainable_variables()])

  def after_create_session(self, session, coord):
    session.run(self.broad_cast_op)

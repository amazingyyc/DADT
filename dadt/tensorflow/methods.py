# coding=utf-8

import re
import ctypes
import platform
import tensorflow as tf

from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

'''
dadt config
cycle_duration_ms: background thread sleep time, millisecond
broad_cast_executor: what kind broadcast executor will be used
0: mpi broad cast
1: nccl broadcast

all_reduce_executor:what kind all reduce executor should be used
0: mpi all reduce
1: nccl all reduce
2: mpi cuda all reduce
'''
class Config(ctypes.Structure):
  _fields_ = [("cycle_duration_ms", ctypes.c_int), 
              ("broad_cast_executor", ctypes.c_int),
              ("all_reduce_executor", ctypes.c_int),
              ("all_reduce_buffer_size", ctypes.c_size_t),
              ("timeline_path", ctypes.c_char_p)]

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

'''set the argumentation type'''
dadt_native_module.init.argtypes = (Config, )

'''
init dadt
broad_cast_executor: accept 'mpi', 'nccl'
all_reduce_executor: accept 'mpi', 'nccl', 'mpicuda'
'''
def init(cycle_duration_ms=5, 
         broad_cast_executor='nccl',
         all_reduce_executor='nccl', 
         all_reduce_buffer_size=67108864,
         timeline_path=None):
  broad_cast_executor = broad_cast_executor.lower()
  all_reduce_executor = all_reduce_executor.lower()

  if 'mpi' == broad_cast_executor:
    broad_cast_executor_type = 0
  elif 'nccl' == broad_cast_executor:
    broad_cast_executor_type = 1
  else:
    raise ValueError('broad_cast_executor must one of "mpi" or "nccl"')

  if 'mpi' == all_reduce_executor:
    all_reduce_executor_type = 0
  elif 'nccl' == all_reduce_executor:
    all_reduce_executor_type = 1
  elif 'mpicuda' == all_reduce_executor:
    all_reduce_executor_type = 2
  else:
    raise ValueError('broad_cast_executor must one of "mpi" or "nccl" or "mpicuda"')

  if timeline_path is None or not isinstance(timeline_path, str) or '' == timeline_path:
    timeline_path_p = None
  else:
    timeline_path_p = ctypes.c_char_p(timeline_path.encode('utf-8'))
  
  config = Config(cycle_duration_ms=cycle_duration_ms,
                    broad_cast_executor=broad_cast_executor_type,
                    all_reduce_executor=all_reduce_executor_type,
                    all_reduce_buffer_size=all_reduce_buffer_size,
                    timeline_path=timeline_path_p)

  dadt_native_module.init(config)

'''shutdown whole system'''
def shutdown():
  dadt_native_module.shutdown()

'''whether have been initialized'''
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
all reduce op, the allreduce is async op, so when firt call this function it will get 0 
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
    assign_ops = [tf.assign(var, broad_cast(var)) for var in tf.trainable_variables()]
    self.broad_cast_op = tf.group(*assign_ops)

  def after_create_session(self, session, coord):
    session.run(self.broad_cast_op)

'''a session hook, will broad cast Global Variable from rank 0 to other ranks'''
class BroadcastGlobalVariablesHook(tf.train.SessionRunHook):
  def __init__(self):
    super(BroadcastGlobalVariablesHook, self).__init__()

    self.broad_cast_op = None

  def begin(self):
    assign_ops = [tf.assign(var, broad_cast(var)) for var in tf.global_variables()]
    self.broad_cast_op = tf.group(*assign_ops)
  
  def after_create_session(self, session, coord):
    session.run(self.broad_cast_op)

'''a optimizer wrapper'''
class DistributedOptimizer(tf.train.Optimizer):
  def __init__(self, optimizer, name=None, use_locking=False, gradient_avg=True):
    self._optimizer = optimizer
    self._gradient_avg = gradient_avg

    if name is None:
      name = 'Distributed{0}'.format(type(optimizer).__name__)

    super(DistributedOptimizer, self).__init__(use_locking=use_locking, name=name)

  def compute_gradients(self, *args, **kwargs):
    '''get the origin gradient'''
    origin_gradients = self._optimizer.compute_gradients(*args, **kwargs)

    if size() > 1:
      dadt_gradients = []

      with tf.name_scope(self._name + 'AllReduce'):
        for grad, var in origin_gradients:
          if grad is not None:
            if isinstance(grad, tf.IndexedSlices):
              grad = tf.convert_to_tensor(grad)
            
            avg_grad = all_reduce(grad)

            if self._gradient_avg:
              dadt_size = tf.cast(size(), dtype=avg_grad.dtype)
              avg_grad  = tf.div(avg_grad, dadt_size)

            dadt_gradients.append((avg_grad, var))
          else:
            dadt_gradients.append((None, var))
        
      return dadt_gradients
    else:
      return origin_gradients

  def apply_gradients(self, *args, **kwargs):
    return self._optimizer.apply_gradients(*args, **kwargs)

  def get_slot(self, *args, **kwargs):
    return self._optimizer.get_slot(*args, **kwargs)

  def get_slot_names(self, *args, **kwargs):
    return self._optimizer.get_slot_names(*args, **kwargs)

  def variables(self, *args, **kwargs):
    return self._optimizer.variables(*args, **kwargs)
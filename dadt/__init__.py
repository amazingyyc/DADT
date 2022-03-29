# # coding=utf-8

# import os, sys, re, ctypes, platform

# '''
# dadt config
# cycle_duration_ms: background thread sleep time, millisecond
# broad_cast_executor: what kind broadcast executor will be used
# 0: mpi broad cast
# 1: nccl broadcast
# 2: mpi cuda boadcast

# all_reduce_executor:what kind all reduce executor should be used
# 0: mpi all reduce
# 1: nccl all reduce
# 2: mpi cuda all reduce
# '''
# class Config(ctypes.Structure):
#   _fields_ = [("cycle_duration_ms", ctypes.c_int),
#               ("broad_cast_executor", ctypes.c_int),
#               ("all_reduce_executor", ctypes.c_int),
#               ("all_reduce_buffer_size", ctypes.c_size_t),
#               ("group_buffer_size", ctypes.c_size_t),
#               ("timeline_path", ctypes.c_char_p)]

# if 'Windows' == platform.system():
#   dadt_lib_suffix = 'dll'
# elif 'Linux' == platform.system():
#   dadt_lib_suffix = 'so'
# elif 'Darwin' == platform.system():
#   dadt_lib_suffix = 'dylib'
# else:
#   raise ValueError('unsupported system')

# '''load dadt library'''
# dadt_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libdadt.{}'.format(dadt_lib_suffix))
# dadt_lib_module = ctypes.CDLL(dadt_lib_path, mode=ctypes.RTLD_GLOBAL)

# '''set the init argumentation type'''
# dadt_lib_module.init.argtypes = (Config, )

# '''
# init dadt
# broad_cast_executor: accept 'mpi', 'nccl', 'mpicuda'
# all_reduce_executor: accept 'mpi', 'nccl', 'mpicuda'
# '''
# def init(cycle_duration_ms=5,
#          broad_cast_executor='nccl',
#          all_reduce_executor='nccl',
#          all_reduce_buffer_size=67108864,
#          group_buffer_size=0,
#          timeline_path=None):
#   broad_cast_executor = broad_cast_executor.lower()
#   all_reduce_executor = all_reduce_executor.lower()

#   if 'mpi' == broad_cast_executor:
#     broad_cast_executor_type = 0
#   elif 'nccl' == broad_cast_executor:
#     broad_cast_executor_type = 1
#   elif 'mpicuda' == broad_cast_executor:
#     broad_cast_executor_type = 2
#   else:
#     raise ValueError('broad_cast_executor must one of "mpi" or "nccl" or "mpicuda"')

#   if 'mpi' == all_reduce_executor:
#     all_reduce_executor_type = 0
#   elif 'nccl' == all_reduce_executor:
#     all_reduce_executor_type = 1
#   elif 'mpicuda' == all_reduce_executor:
#     all_reduce_executor_type = 2
#   else:
#     raise ValueError('broad_cast_executor must one of "mpi" or "nccl" or "mpicuda"')

#   if timeline_path is None or not isinstance(timeline_path, str) or '' == timeline_path:
#     timeline_path_p = None
#   else:
#     timeline_path_p = ctypes.c_char_p(timeline_path.encode('utf-8'))

#   config = Config(cycle_duration_ms=cycle_duration_ms,
#                   broad_cast_executor=broad_cast_executor_type,
#                   all_reduce_executor=all_reduce_executor_type,
#                   all_reduce_buffer_size=all_reduce_buffer_size,
#                   group_buffer_size=group_buffer_size,
#                   timeline_path=timeline_path_p)

#   dadt_lib_module.init(config)

#   '''shutdown whole system'''
# def shutdown():
#   dadt_lib_module.shutdown()

# '''whether have been initialized'''
# def initialized():
#   return dadt_lib_module.initialized()

# '''how many process'''
# def size():
#   return dadt_lib_module.size()

# '''how many process in current machine'''
# def local_size():
#   return dadt_lib_module.local_size()

# '''the rank of current process'''
# def rank():
#   return dadt_lib_module.rank()

# '''local rank'''
# def local_rank():
#   return dadt_lib_module.local_rank()

# '''barrier all process'''
# def barrier():
#   dadt_lib_module.barrier()

# '''local barrier all process'''
# def local_barrier():
#   dadt_lib_module.local_barrier()
# coding=utf-8

import os
import re
import sys
import platform
import subprocess

import tensorflow as tf

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

'''
python setup.py build
python setup.py install
python setup.py sdist
python setup.py bdist_wininst
python setup.py bdist_rpm
'''

def find_in_path(name, path):
  '''Find a file in a search path'''

  for dir in path.split(os.pathsep):
    bin_path = os.path.join(dir, name)

    if os.path.exists(bin_path):
      return os.path.abspath(bin_path)

  return None

'''find a file in folder'''
def find_file_in_folder(folder, name):
  for file_name in os.listdir(folder):
    file_path = os.path.join(folder, file_name)

    if os.path.isfile(file_path):
      if file_name == name:
        return file_path
    elif os.path.isdir(file_path):
      correct_path = find_file_in_folder(file_path, name)

      if None != correct_path:
        return correct_path

  return None

def find_file_in_folders(folders, name):
  for folder in folders:
    correct_path = find_file_in_folder(folder, name)

    if None != correct_path:
      return correct_path

  return None

def find_tensorflow_lib_paths(folder):
  lib_paths = []

  for file_name in os.listdir(folder):
    if os.path.isfile(os.path.join(folder, file_name)):
      if file_name.endswith('.so') or file_name.endswith('.so.1'):
        lib_paths.append(os.path.join(folder, file_name))

  if 0 == len(lib_paths):
    raise ValueError('can not find tensorflow libs in folder:' + folder)

  return lib_paths

class CMakeExtension(Extension):
  def __init__(self, name, build_for_nccl = False, cmake_dir=''):
    Extension.__init__(self, name, sources=[])

    self.build_for_nccl = build_for_nccl
    self.cmake_dir = os.path.abspath(cmake_dir)

class CMakeBuildExt(build_ext):
  def run(self):
    try:
      out = subprocess.check_output(['cmake', '--version'])
    except OSError:
      raise RuntimeError('CMake must be installed to build the following extensions: ' +
                            ', '.join(e.name for e in self.extensions))

    if platform.system() == 'Windows':
      cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
      if cmake_version < '3.8.0':
          raise RuntimeError('CMake >= 3.8.0 is required on Windows')

    for ext in self.extensions:
      self.build_extension(ext)

  def build_extension(self, ext):
    lib_suffix = ['.framework', '.o', '.so', '.a', '.dylib']

    ext_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
    ext_dir = os.path.join(ext_dir, ext.name)

    print('ext_dir:', ext_dir)

    '''
    tensorflow include path: TENSORFLOW_INCLUDE_DIRS
    tf libs: TENSORFLOW_LIB_PATHS

    cuda include path: CUDA_INCLUDE_DIRS
    cuda lib: CUDA_LIB_PATHS

    nccl include path: NCCL_INCLUDE_DIRS
    nccl lib: NCCL_LIB_PATHS
    '''

    # cmake args
    # cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + ext_dir, '-DPYTHON_EXECUTABLE=' + sys.executable]
    cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + ext_dir]

    # check gcc version whether add -D_GLIBCXX_USE_CXX11_ABI=0 when build.
    try:
      gcc_out = subprocess.check_output(['gcc', '-v'], stderr=subprocess.STDOUT)
    except OSError:
      raise RuntimeError('get gcc version get error!')
    print("gcc_out.decode()", gcc_out.decode())
    if 'Linux' == platform.system():
      gcc_version = LooseVersion(re.search(r'version\s*([\d.]+)', gcc_out.decode()).group(1))

      if gcc_version >= '5':
        cmake_args.append('-DADD_GLIBCXX_USE_CXX11_ABI=1')

    # add tensorflow args
    cmake_args.append('-DTENSORFLOW_INCLUDE_DIRS=' + tf.sysconfig.get_include())

    # tf lib folder
    tf_lib_folder = tf.sysconfig.get_lib()

    tf_lib_paths = find_tensorflow_lib_paths(tf_lib_folder)

    # set tensorflow lib
    cmake_args.append('-DTENSORFLOW_LIB_PATHS=' + ';'.join(tf_lib_paths))

    print('TENSORFLOW_INCLUDE_DIRS:', tf.sysconfig.get_include())
    print('TENSORFLOW_LIB_PATHS:', ';'.join(tf_lib_paths))

    if ext.build_for_nccl:
      if 'Linux' != platform.system():
        raise ValueError('build_for_nccl only works for Linux.')

      # set NCCL build flag
      cmake_args.append('-DHAVE_NCCL=1')

      # add CUDA args
      nvcc_path = find_in_path('nvcc', os.environ['PATH'])

      if nvcc_path is None:
        raise ValueError('Build for GPU, but can not find nvcc, add it to path try again')

      cuda_home = os.path.dirname(os.path.dirname(nvcc_path))

      cmake_args.append('-DCUDA_INCLUDE_DIRS=' + os.path.join(cuda_home, 'include') + ';/usr/local')

      cuda_lib_folder = os.path.join(cuda_home, 'lib64')

      cuda_lib_paths = [os.path.join(cuda_lib_folder, file) \
                          for file in os.listdir(cuda_lib_folder) \
                            if os.path.isfile(os.path.join(cuda_lib_folder, file)) and \
                              os.path.splitext(file)[-1] in lib_suffix]

      cmake_args.append('-DCUDA_LIB_PATHS=' + ';'.join(cuda_lib_paths))

      print('CUDA_INCLUDE_DIRS:', os.path.join(cuda_home, 'include'))
      print('CUDA_LIB_PATHS:', ';'.join(cuda_lib_paths))

      # add NCCL args
      nccl_header_search_folders = ['/usr/include', '/usr/local/include']

      if 'NCCL_DIR' is os.environ.keys():
        nccl_header_search_folders.append(os.environ['NCCL_DIR'] + '/include')

      nccl_header_file_path = find_file_in_folders(nccl_header_search_folders, 'nccl.h')

      if nccl_header_file_path is None:
        raise ValueError('Can not find nccl.h file, please set NCC_DIR environment and try again')

      cmake_args.append('-DNCCL_INCLUDE_DIRS=' + os.path.dirname(nccl_header_file_path))

      nccl_lib_search_folders = ['/lib', '/lib64', '/usr/lib', '/usr/lib64', '/usr/local/lib', '/usr/local/lib64']

      if 'NCCL_DIR' in os.environ.keys():
        nccl_lib_search_folders.append(os.environ['NCCL_DIR'] + '/lib')

      nccl_lib_file_path = find_file_in_folders(nccl_lib_search_folders, 'libnccl.so')

      if nccl_lib_file_path is None:
        raise ValueError('Can not find libnccl.so file, please set NCC_DIR environment and try again')

      cmake_args.append('-DNCCL_LIB_PATHS=' + nccl_lib_file_path)

      print('NCCL_INCLUDE_DIRS:', os.path.dirname(nccl_header_file_path))
      print('NCCL_LIB_PATHS:', nccl_lib_file_path)

    cfg = 'Debug' if self.debug else 'Release'
    build_args = ['--config', cfg]

    if platform.system() == 'Windows':
      cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), ext_dir)]

      if sys.maxsize > 2**32:
        cmake_args += ['-A', 'x64']

        build_args += ['--', '/m']
        build_args += ['/p:PreferredToolArchitecture=x64']
      else:
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j8']

    env = os.environ.copy()
    env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''), self.distribution.get_version())

    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)

    print('build_temp:', self.build_temp)
    print('cmake_args:', cmake_args)
    print('build_args:', build_args)

    subprocess.check_call(['cmake', ext.cmake_dir] + cmake_args, cwd=self.build_temp, env=env)
    subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='DADT',
    version='0.1.0',
    description='A Decentrilized Asynchronously Distribute Training framwork',
    long_description='',
    author='Yan Yuanchi',
    author_email='amazingyyc@outlook.com',
    zip_safe=False,
    packages=['dadt', 'dadt/tensorflow'],
    ext_modules=[CMakeExtension(name='dadt', build_for_nccl=True, cmake_dir='.')],
    cmdclass={'build_ext' : CMakeBuildExt},
)
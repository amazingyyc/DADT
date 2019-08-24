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

class CMakeExtension(Extension):
  def __init__(self, name, build_for_gpu = False, cmake_dir=''):
    Extension.__init__(self, name, sources=[])

    self.build_for_gpu = build_for_gpu
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
    '''

    # cmake args
    cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + ext_dir, '-DPYTHON_EXECUTABLE=' + sys.executable]

    # add tensorflow args
    cmake_args.append('-DTENSORFLOW_INCLUDE_DIRS=' + tf.sysconfig.get_include())
        
    #-D_GLIBCXX_USE_CXX11_ABI=0
    cmake_args.append('-D_GLIBCXX_USE_CXX11_ABI=0')

    # tf lib folder
    tf_lib_folder = tf.sysconfig.get_lib()

    print('tf_lib_folder:', tf_lib_folder)

    tf_lib_paths = [os.path.join(tf_lib_folder, file) \
                        for file in os.listdir(tf_lib_folder) \
                        if os.path.isfile(os.path.join(tf_lib_folder, file)) and \
                           os.path.splitext(file)[-1] in lib_suffix]
    # set tensorflow lib
    cmake_args.append('-DTENSORFLOW_LIB_PATHS=' + ';'.join(tf_lib_paths))
    
    if ext.build_for_gpu:
      # set GPU build flag
      cmake_args.append('-DHAVE_CUDA=1')

      # add CUDA args
      nvcc_path = find_in_path('nvcc', os.environ['PATH'])

      if nvcc_path is None:
        raise ValueError('Build for GPU, but can not find nvcc, add it to path try again')
      
      cuda_home = os.path.dirname(os.path.dirname(nvcc_path))

      if platform.system() == 'Linux' or platform.system() == 'Darwin':
        cmake_args.append('-DCUDA_INCLUDE_DIRS=' + os.path.join(cuda_home, 'include') + ';/usr/local')
      else:
        cmake_args.append('-DCUDA_INCLUDE_DIRS=' + os.path.join(cuda_home, 'include'))

      cuda_lib_folder = os.path.join(cuda_home, 'lib64')

      cuda_lib_paths = [os.path.join(cuda_lib_folder, file) \
                          for file in os.listdir(cuda_lib_folder) \
                            if os.path.isfile(os.path.join(cuda_lib_folder, file)) and \
                              os.path.splitext(file)[-1] in lib_suffix]
      
      cmake_args.append('-DCUDA_LIB_PATHS=' + ';'.join(cuda_lib_paths))
    
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
    packages=['dadt/tensorflow'],
    ext_modules=[CMakeExtension(name='dadt', build_for_gpu=False, cmake_dir='.')],
    cmdclass={'build_ext' : CMakeBuildExt},
)









# coding=utf-8

import tensorflow as tf

from dadt.tensorflow.ops import init
from dadt.tensorflow.ops import shutdown
from dadt.tensorflow.ops import initialized
from dadt.tensorflow.ops import size
from dadt.tensorflow.ops import local_size
from dadt.tensorflow.ops import rank
from dadt.tensorflow.ops import local_rank
from dadt.tensorflow.ops import barrier
from dadt.tensorflow.ops import local_barrier
from dadt.tensorflow.ops import broad_cast
from dadt.tensorflow.ops import BroadcastTrainableVariablesHook
#coding=utf-8

import tensorflow as tf
import dadt.tensorflow as dadt

dadt.init()

if 0 == dadt.rank():
  var = tf.ones([2, 3], name='var')
else:
  var = tf.ones([2, 3], name='var')

b_var = dadt.broad_cast(var)
p_var = tf.Print(b_var)

with td.Session() as sess:
  sess.run(p_var)

dadt.shutdown()
#coding=utf-8

import tensorflow as tf
import dadt.tensorflow as dadt

dadt.init()

if 0 == dadt.rank():
  var = tf.zeros([2, 3], name='var')
else:
  var = tf.ones([2, 3], name='var')

b_var = dadt.broad_cast(var)
p_var = tf.print(b_var)

with tf.Session() as sess:
  sess.run(p_var)

dadt.shutdown()
#coding=utf-8

import tensorflow as tf
import dadt.tensorflow as dadt

dadt.init()

if 0 == dadt.rank():
  var1 = tf.ones([2, 3], name='var1')
  var2 = tf.ones([2, 3], name='var1')
else:
  var1 = tf.zeros([2, 3], name='var1')
  var2 = tf.zeros([2, 3], name='var1')

p1 = tf.print(dadt.all_reduce(var1))
p2 = tf.print(dadt.all_reduce(var2))

with tf.Session() as sess:
  sess.run([p1, p2])

with tf.Session() as sess:
  sess.run([p1, p2])

dadt.shutdown()
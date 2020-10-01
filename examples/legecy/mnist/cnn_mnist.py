# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import dadt.tensorflow as dadt

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    optimizer = dadt.DistributedOptimizer(optimizer)

    tvars = tf.trainable_variables()
    grad_vars = optimizer.compute_gradients(loss, tvars)
    (grads, _) = zip(*grad_vars)

    (grads, l2_norm) = tf.clip_by_global_norm(grads, clip_norm=1.0)
    
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.train.get_global_step())

    # train_op = optimizer.minimize(
    #     loss=loss,
    #     global_step=tf.train.get_global_step())

    global_step = tf.train.get_or_create_global_step()
    log_hook = tf.train.LoggingTensorHook({"loss:": loss, "global_step:": global_step, "l2_norm": l2_norm}, every_n_iter=10)

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[log_hook])

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, 
          predictions=predictions["classes"])}

  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
  '''init dadt'''
  dadt.init(cycle_duration_ms=3,
            broad_cast_executor='nccl',
            all_reduce_executor='nccl')

  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data   = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data    = mnist.test.images  # Returns np.array
  eval_labels  = np.asarray(mnist.test.labels, dtype=np.int32)
  
  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True
  session_config.gpu_options.visible_device_list = str(dadt.local_rank())

  run_config = tf.estimator.RunConfig(
    model_dir='model' + str(dadt.local_rank()),
    session_config=session_config,
    train_distribute=None)

  mnist_classifier = tf.estimator.Estimator(
    model_dir='model' + str(dadt.local_rank()),
    model_fn=cnn_model_fn,
    config=run_config)

  def train_input_fn(params):
    """An input function for training"""
    dataset = tf.data.Dataset.from_tensor_slices(({"x": train_data}, train_labels))

    dataset = dataset.shard(dadt.size(), dadt.rank()).shuffle(100).repeat().batch(100)

    return dataset

  '''create a broad cast hook'''
  dadt_hook = dadt.BroadcastTrainableVariablesHook()

  mnist_classifier.train(
    input_fn=train_input_fn,
    steps=40000,
    hooks=[dadt_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

  '''shut down dadt'''
  dadt.shutdown()

if __name__ == "__main__":
  tf.app.run()


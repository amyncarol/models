
"""Builds the convolutional neutral network for crystal graph.

Adapted from tensorflow tutorial cifar10

Summary of available functions:

 # Compute input crystal graphs and targeted property for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, ground_truth = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the ground_truth.
 loss = loss(predictions, ground_truth)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys

import tensorflow as tf
import cnn_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 100,
                            """Number of structures to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/Users/yao/Google Drive/models/data/formation_energy',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', True,
                            """Train the model using fp16.""")

# Global constants describing the data set.
MAX_N_SITES = cnn_input.MAX_N_SITES       #the maximum of number of sites among all structures
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 250


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 10.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001       # Initial learning rate.


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  tensor_name = x.op.name
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def inputs(eval_data):
  """Generate input data for training or evaluation

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns: 
    energies: tensor of shape (batch_size, 1)
    sites_matrices: tensor of shape (batch_size, max_n_sites, feature_length)
    adj_matrices: tensor of shape (batch_size, max_n_sites, max_n_sites)

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  energies_file = os.path.join(FLAGS.data_dir, 'energies.npy')
  sites_matrices_file = os.path.join(FLAGS.data_dir, 'sites_matrices.npy')
  adj_matrices_file = os.path.join(FLAGS.data_dir, 'adj_matrices.npy')

  training_set, eval_set = cnn_input.get_training_eval(energies_file, sites_matrices_file, adj_matrices_file)

  energies, sites_matrices, adj_matrices = \
      cnn_input.inputs(eval_data = eval_data, 
                      batch_size = FLAGS.batch_size,  
                      training_set = training_set,
                      eval_set = eval_set)

  if FLAGS.use_fp16:
    energies = tf.cast(energies, tf.float16)
    sites_matrices = tf.cast(sites_matrices, tf.float16)
    adj_matrices = tf.cast(adj_matrices, tf.float16)
  return energies, sites_matrices, adj_matrices


def inference(sites_matrices, adj_matrices):
  """Build the crystal graph cnn model.

  Args:
    These are tensors returned by inputs()

    sites_matrices: tensor of shape (batch_size, max_n_sites, feature_length)
    adj_matrices: tensor of shape (batch_size, max_n_sites, max_n_sites)

  Returns:
    output: the predicted energies
  """
  
  #conv1
  with tf.variable_scope('conv1') as scope:

    ##prepare concated twisted matrix sites_matrices_4d
    max_n_sites = sites_matrices.shape.as_list()[1]
    sites_matrices_4d_1 = tf.tile(tf.expand_dims(sites_matrices, axis = 1), [1, max_n_sites, 1, 1])
    sites_matrices_4d_2 = tf.tile(tf.expand_dims(sites_matrices, axis = 2), [1, 1, max_n_sites, 1])
    sites_matrices_4d = tf.concat([sites_matrices_4d_1, sites_matrices_4d_2], 3)

    ##prepare the adj_matrices_4d
    feature_length = sites_matrices.shape.as_list()[2]
    adj_matrices = tf.cast(adj_matrices, sites_matrices.dtype)
    adj_matrices_4d = tf.tile(tf.expand_dims(adj_matrices, axis = 3), [1, 1, 1, feature_length])

    ##the sigmoid part
    weights_f = _variable_with_weight_decay('weights_f',
                                         shape=[feature_length*2, feature_length],
                                         stddev=5e-2,
                                         wd=None)
    biases_f = _variable_on_cpu('biases_f', [feature_length], tf.constant_initializer(0.0))
    pre_sigmoid = tf.nn.bias_add(tf.tensordot(sites_matrices_4d, weights_f, axes = [[3], [0]]), biases_f)
    sigmoid = tf.sigmoid(pre_sigmoid, name='sigmoid')

    ##the activation part
    weights_s = _variable_with_weight_decay('weights_s',
                                         shape=[feature_length*2, feature_length],
                                         stddev=5e-2,
                                         wd=None)
    biases_s = _variable_on_cpu('biases_s', [feature_length], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(tf.tensordot(sites_matrices_4d, weights_s, axes = [[3], [0]]), biases_s)
    activation = tf.nn.relu(pre_activation, name='activation')

    ##element-wise multiplication of sigmoid and activation
    el_multiply = tf.multiply(sigmoid, activation)
    connectivity_weighted = tf.multiply(el_multiply, adj_matrices_4d)
    weighted_sum = tf.reduce_sum(connectivity_weighted, axis = 1)

    conv1 = tf.add(weighted_sum, sites_matrices, name=scope.name)
    _activation_summary(conv1)

  #pool1
  pool1 = tf.reduce_mean(conv1,  axis = 1, name='pool1')

  #hidden layer
  with tf.variable_scope('hidden1') as scope: 
    dim = pool1.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 15],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [15], tf.constant_initializer(0.1))
    hidden1 = tf.nn.relu(tf.matmul(pool1, weights) + biases, name=scope.name)
    _activation_summary(hidden1)

  #output layer
  with tf.variable_scope('output') as scope: 
    dim = hidden1.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 1],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [1], tf.constant_initializer(0.1))
    output = tf.nn.bias_add(tf.matmul(hidden1, weights), biases, name=scope.name)
   
  return output
  

def loss(energies_hat, energies):
  """Add L2Loss to all the trainable variables.

  Args:
    energies_hat: output from inference().
    energies: energies from inputs(). 1-D tensor of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average loss across the batch.
  l2_loss = tf.nn.l2_loss(energies_hat-energies, name='l2_loss')
  n_batch = tf.cast(energies.shape[0].value, dtype=l2_loss.dtype)
  l2_loss_mean = tf.divide(l2_loss, n_batch, name='l2_loss_mean')
  tf.add_to_collection('losses', l2_loss_mean)

  # The total loss is defined as the L2 loss plus all of the weight
  # decay terms (also L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().

  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train the model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

if __name__=='__main__':
  sites_matrices = tf.constant([[[  1.00000000e+00,   2.00000000e+00,   3.00000000e+00],
      [  4.00000000e+00,   5.00000000e+00,   6.00000000e+00]],

     [[  1.00000000e-01,   2.00000000e-01,   3.00000000e-01],
      [  4.00000000e-01,   5.00000000e-01,   6.00000000e-01]],

     [[  1.00000000e-02,   2.00000000e-02,   3.00000000e-02],
      [  4.00000000e-02,   5.00000000e-02,   6.00000000e-02]],

     [[  1.00000000e-03,   2.00000000e-03,   3.00000000e-03],
      [  4.00000000e-03,   5.00000000e-03,   6.00000000e-03]]])
  adj_matrices = tf.constant([[[0, 1],
        [1, 0]],

       [[0, 1],
        [1, 0]],

       [[0, 2],
        [2, 0]],

       [[0, 3],
        [3, 0]]])
  energies = tf.constant([0, 1, 2, 3], dtype=tf.float32)

  with tf.Session() as sess:
    energies_hat = inference(sites_matrices, adj_matrices)
    result = loss(energies_hat, energies)
    sess.run(tf.global_variables_initializer())
    print(sess.run(result))


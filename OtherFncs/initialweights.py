import tensorflow as tf
import numpy as np
import math, random

"""WEIGHTS Initializer"""


def initial_weights(iweight, shape, weights_path = None, session = None):  # session = None, weights_path = None):

    initializer = 'None'

    if iweight == 'constant':
        init_val = random.uniform(0.1,254.9)
        initializer = tf.constant_initializer(init_val)

    elif iweight == 'runiform':
        minval = random.uniform(0.1,125)
        maxval = random.uniform(125.1,254.9)
        initializer = tf.random_uniform_initializer(minval, maxval)

    elif iweight == 'rnormal':
        mean = random.uniform(0.1,254.9)
        stddev = random.uniform(0.1,254.9)
        initializer = tf.random_normal_initializer(mean, stddev)

    elif iweight == 'xavier':
        fan_in, fan_out = get_dims(shape)
        spread = math.sqrt(6.0 / (fan_in + fan_out))
        initializer = tf.random_uniform_initializer(-spread, spread)

    elif iweight == 'tnormal':
        mean = random.uniform(0.1,254.9)
        stddev = random.uniform(0.1,254.9)
        initializer = tf.truncated_normal_initializer(mean, stddev)

    elif iweight == 'load' and session != None:
        weights_dict = np.load(weights_path, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:
            with tf.variable_scope(op_name, reuse=True):

                # Assign weights/biases to their corresponding tf variable
                for data in weights_dict[op_name]:

                    # Biases
                    if len(data.shape) == 1:
                        var = tf.get_variable('biases', trainable=False)
                        session.run(var.assign(data))

                    # Weights
                    else:
                        var = tf.get_variable('weights', trainable=False)
                        session.run(var.assign(data))

    else:
        init_val = random.uniform(0.1,254.9)  # input('Input Initial Value')
        initializer = tf.constant_initializer(init_val)

    return initializer


def get_dims(shape):
    fan_in = np.prod(shape[:-1])
    fan_out = shape[-1]
    return fan_in, fan_out

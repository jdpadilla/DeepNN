import tensorflow as tf
import numpy as np
# from DeepNN2_0.Models.models import Model
# import types

# from DeepNN2_0.Models.Configs.alexnetconfig import *
from DeepNN2_0.OtherFncs.initialweights import initial_weights
from DeepNN2_0.OtherFncs.initialbias import initial_bias
# from DeepNN2_0.LayerOps.convolution import Convolution
# from DeepNN2_0.LayerOps.normalizations import Normalization
# from DeepNN2_0.LayerOps.pooling import Pooling
# from DeepNN2_0.LayerOps.dense import Dense
# from DeepNN2_0.LayerOps.dropout import Dropout


class AlexNet(object):

    """
    Use the parameters proposed in http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
    https://github.com/dmlc/minerva/wiki/Walkthrough:-AlexNet
    """

    def __init__(self, x, args):
        # super().__init__(x, args)
        self.x = x
        self.args = args
        self.build()

    def build(self):

        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
        conv1 = conv(self.x, 11, 11, 96, 4, 4, padding='VALID', name='convL1')
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='poolL1')
        norm1 = lrn(pool1, 2, 2e-05, 0.75, name='normL1')

        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        conv2 = conv(norm1, 5, 5, 256, 1, 1, groups=2, name='convL2')
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='poolL2')
        norm2 = lrn(pool2, 2, 2e-05, 0.75, name='normL2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(norm2, 3, 3, 384, 1, 1, name='convL3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='convL4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='convL5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='poolL5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fcL6')
        dropout6 = dropout(fc6, self.args.dropout_rate)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name='fcL7')
        dropout7 = dropout(fc7, self.args.dropout_rate)

        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        self.fc8 = fc(dropout7, 4096, self.args.num_classes, relu=False, name='fcL8')

    def load_initial_weights(self, session):
        """
        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ come
        as a dict of lists (e.g. weights['conv1'] is a list) and not as dict of
        dicts (e.g. weights['conv1'] is a dict with keys 'weights' & 'biases') we
        need a special load function
        """

        # Load the weights into memory
        weights_dict = np.load(self.args.wpath, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if the layer is one of the layers that should be reinitialized
            if op_name not in self.args.train_layers:

                with tf.variable_scope(op_name, reuse=True):

                    # Loop over list of weights/biases and assign them to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:

                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))

                        # Weights
                        else:

                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))

    """
    Predefine all necessary layer for the AlexNet
    """

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """
    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights',
                                  shape=[filter_height, filter_width, input_channels / groups, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

        if groups == 1:
            conv = convolve(x, weights)

        # In the cases of multiple groups, split inputs & weights and
        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(axis=3, values=output_groups)

        # Add biases
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)

        return relu

def fc(x, num_in, num_out, name, relu=True):
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu is True:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)

def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha,
                                              beta=beta, bias=bias, name=name)

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)

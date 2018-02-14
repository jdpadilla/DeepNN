import tensorflow as tf
from DeepNN2_0.Models.models import Model

from DeepNN2_0.Models.Configs.alexnetconfig import *

from DeepNN2_0.LayerOps.convolution import Convolution
from DeepNN2_0.LayerOps.normalizations import Normalization
from DeepNN2_0.LayerOps.pooling import Pooling
from DeepNN2_0.LayerOps.dense import Dense
from DeepNN2_0.LayerOps.dropout import Dropout


class AlexNet(Model):

    """
    Use the parameters proposed in http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
    https://github.com/dmlc/minerva/wiki/Walkthrough:-AlexNet
    """

    def __init__(self, x, args):
        super().__init__(x, args)
        # self.x = x
        # self.args = args
        self.build()


    def build(self):

        # Layer 1
        ops_args = alexnetargs1(self.args.iweight, self.args.ibias)
        # conv1 = Convolution.conv2d(self.x, ops_args.convargs, 'convL1')
        conv1 = conv(self.x, 11, 11, 96, 4, 4, padding='VALID', name='convL1')
        pool1 = Pooling.maxp(conv1, ops_args.poolargs, 'poolL1')
        norm1 = Normalization.lrn(pool1, ops_args.normargs, 'normL1')

        # Layer 2
        ops_args = alexnetargs2(self.args.iweight, self.args.ibias)
        # conv2 = Convolution.conv2d(norm1, ops_args.convargs, 'convL2')
        conv2 = conv(norm1, 5, 5, 256, 1, 1, groups=2, name='convL2')
        pool2 = Pooling.maxp(conv2, ops_args.poolargs, 'poolL2')
        norm2 = Normalization.lrn(pool2, ops_args.normargs, 'normL2')

        # Layer 3
        ops_args = alexnetargs3(self.args.iweight, self.args.ibias)
        # conv3 = Convolution.conv2d(norm2, ops_args.convargs, 'convL3')
        conv3 = conv(norm2, 3, 3, 384, 1, 1, name='convL3')

        # Layer 4
        ops_args = alexnetargs4(self.args.iweight, self.args.ibias)
        # conv4 = Convolution.conv2d(conv3, ops_args.convargs, 'convL4')
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='convL4')

        # Layer 5
        ops_args = alexnetargs5(self.args.iweight, self.args.ibias)
        # conv5 = Convolution.conv2d(conv4, ops_args.convargs, 'convL5')
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='convL5')
        pool5 = Pooling.maxp(conv5, ops_args.poolargs, 'poolL5')

        # Layer 6
        ops_args = alexnetargs6(self.args.iweight, self.args.ibias)
        flattened = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = Dense.fc(flattened, ops_args.fcargs, 'fcL6')
        dropout6 = Dropout.drop(fc6, self.args.dropout_rate, 'dropL6')

        # Layer 7
        ops_args = alexnetargs7(self.args.iweight, self.args.ibias)
        fc7 = Dense.fc(dropout6, ops_args.fcargs, 'fcL7')
        dropout7 = Dropout.drop(fc7, self.args.dropout_rate, 'dropL7')

        # Layer 8
        ops_args = alexnetargs8(self.args.num_classes, self.args.iweight, self.args.ibias)
        self.fc8 = Dense.fc(dropout7, ops_args.fcargs, 'fcL8')


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
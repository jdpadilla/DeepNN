import tensorflow as tf
from DeepNN2_0.LayerOps.nnetops import NNetOps

from DeepNN2_0.OtherFncs.initialweights import initial_weights
from DeepNN2_0.OtherFncs.initialbias import initial_bias

from DeepNN2_0.LayerOps.activations import Activations


class Convolution(NNetOps):

    def sepconv(self, args, name):
        pass

    # @staticmethod
    # def conv1d(x, args, name):
    #     input_channels = int(x.get_shape()[-1])
    #
    #
    #     convolve = lambda i, k: tf.nn.conv1d(i, k,
    #                                          stride = args.stride,
    #                                          padding =args.padding)
    #
    #     with tf.variable_scope(name) as scope:
    #         # Create tf variables for the weights and biases of the conv layer
    #         weights = tf.get_variable('weights', shape=[args.filter_height,
    #                                                     args.filter_width,
    #                                                     input_channels,
    #                                                     args.num_filters])
    #         biases = tf.get_variable('biases', shape=[args.num_filters])
    #
        #     conv = convolve(x, weights)
        #
        #     # Add biases
        #     bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
        #
        #     # Apply activation function
        #     result = Activations.actv(bias, args.actv, name=scope.name)

            # return result

    @staticmethod
    def conv2d(x, args, name):

        input_channels = int(x.get_shape()[-1])

        iweight = initial_weights(args.iweight, x)
        ibias = initial_bias(args.ibias, x)

        convolve = lambda i, k: tf.nn.conv2d(i, k,
                                             strides=[1, args.stride_y, args.stride_x, 1],
                                             padding=args.padding)

        with tf.variable_scope(name) as scope:
            # Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable(name='weights',
                                      initializer=iweight,
                                      shape=[args.filter_height, args.filter_width,
                                             input_channels/args.groups, args.num_filters],
                                      trainable=True)

            biases = tf.get_variable(name='biases',
                                     initializer=ibias,
                                     shape=[args.num_filters],
                                     trainable=True)

            if args.groups == 1:
                conv = convolve(x,weights)

            else:
                input_groups = tf.split(axis=3, num_or_size_splits=args.groups, value=x)
                weight_groups = tf.split(axis=3, num_or_size_splits=args.groups, value=weights)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

                # Concat the convolved output
                conv = tf.concat(axis=3, values=output_groups)

            # Add biases
            bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

            # Apply activation function
            result = Activations.actv(bias, args.actv, name=scope.name)

            return result

    # @staticmethod
    # def conv3d(x, args, name):
    #     input_channels = int(x.get_shape()[-1])
    #
    #     # Create lambda function for the convolution
    #     convolve = lambda i, k: tf.nn.conv3d(i, k,
    #                                          strides=[1, args.stride_z, args.stride_y, args.stride_x, 1],
    #                                          padding=args.padding)
    #
    #     with tf.variable_scope(name) as scope:
    #         # Create tf variables for the weights and biases of the conv layer
    #         weights = tf.get_variable('weights', shape=[args.filter_height,
    #                                                     args.filter_width,
    #                                                     input_channels / args.groups,
    #                                                     args.num_filters])
    #         biases = tf.get_variable('biases', shape=[args.num_filters])
    #
        #     conv = convolve(x, weights)
        #
        #     # Add biases
        #     bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
        #
        #     # Apply activation function
        #     result = Activations.actv(bias, args.actv, name=scope.name)
        #
        #     return result
    #
    # @staticmethod
    # def deconv2d(x, args,name): # REVISE Since tf.nn.conv2d_transpose is its backward counterpart
    #     input_channels = int(x.get_shape()[-1])
    #
    #     # Create lambda function for the convolution
    #     deconvolve = lambda i, k: tf.nn.conv2d_transpose(i, k, output_shape= 'None',
    #                                                      strides=[1,
    #                                                               args.stride_z,
    #                                                               args.stride_y,
    #                                                               args.stride_x,
    #                                                               1],
    #                                                      padding=args.padding)
    #
    #     with tf.variable_scope(name) as scope:
    #         # Create tf variables for the weights and biases of the conv layer
    #         weights = tf.get_variable('weights', shape=[args.filter_height,
    #                                                     args.filter_width,
    #                                                     input_channels / args.groups,
    #                                                     args.num_filters])
    #         biases = tf.get_variable('biases', shape=[args.num_filters])
    #
        #     conv = deconvolve(x, weights)
        #
        #     # Add biases
        #     bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
        #
        #     # Apply activation function
        #     result = Activations.actv(bias, args.actv, name=scope.name)
        #
        #     return result
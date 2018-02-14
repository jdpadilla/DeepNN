import tensorflow as tf
from DeepNN2_0.LayerOps.nnetops import NNetOps


# POOLING Ops
class Pooling(NNetOps):

    @staticmethod
    def avgp(x, args, name):
        return tf.nn.avg_pool(x, ksize=[1, args.filter_height, args.filter_width, 1],
                              strides=[1, args.stride_y, args.stride_x, 1],
                              padding=args.padding, name=name)

    @staticmethod
    def maxp(x, args, name):
        return tf.nn.max_pool(x, ksize=[1, args.filter_height, args.filter_width, 1],
                              strides=[1, args.stride_y, args.stride_x, 1],
                              padding=args.padding, name=name)

    @staticmethod
    def argmaxp(x, args, name):
        return tf.nn.max_pool_with_argmax(x, ksize=[1, args.filter_height, args.filter_width, 1],
                              strides=[1, args.stride_y, args.stride_x, 1],
                              padding=args.padding, name=name)

    @staticmethod
    def avg3d(x, args, name):
        return tf.nn.avg_pool3d(x, ksize=[1, args.filter_height, args.filter_width, 1],
                              strides=[1, args.stride_y, args.stride_x, 1],
                              padding=args.padding, name=name)

    @staticmethod
    def max3d(x, args, name):
        return tf.nn.max_pool3d(x, ksize=[1, args.filter_height, args.filter_width, 1],
                              strides=[1, args.stride_y, args.stride_x, 1],
                              padding=args.padding, name=name)

    @staticmethod
    def fr_avg(x, args, name):
        pass

    @staticmethod
    def fr_max(x, args, name):
        pass

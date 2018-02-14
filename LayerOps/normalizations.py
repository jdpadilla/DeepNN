import tensorflow as tf
from DeepNN2_0.LayerOps.nnetops import NNetOps

# NORMALIZATION
class Normalization(NNetOps):

    # tf.nn.l2_normalize
    def l2norm(self):
        pass
    # tf.nn.local_response_normalization
    @staticmethod
    def lrn(x, args, name):
        return tf.nn.local_response_normalization(x, depth_radius=args.radius, alpha=args.alpha,
                                                  beta=args.beta, bias=args.bias, name=name)
    # tf.nn.sufficient_statistics
    def suffstats(self):
        pass
    # tf.nn.normalize_moments
    def normmoments(self):
        pass
    # tf.nn.moments
    def moments(self):
        pass
    # tf.nn.weighted_moments
    def wmoments(self):
        pass
    # tf.nn.fused_batch_norm
    def fbatchnorm(self):
        pass
    # tf.nn.batch_normalization
    def batchnorm(self):
        pass
    # tf.nn.batch_norm_with_global_normalization
    def batchnormwgnorm(self):
        pass

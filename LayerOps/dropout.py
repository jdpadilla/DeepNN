import tensorflow as tf
from DeepNN2_0.LayerOps.nnetops import NNetOps

class Dropout(NNetOps):

    @staticmethod
    def drop(x, args, name):
        return tf.nn.dropout(x, args, name=name)

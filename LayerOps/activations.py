import tensorflow as tf
from DeepNN2_0.LayerOps.nnetops import NNetOps

# Activation Functions
class Activations(NNetOps):

    @staticmethod
    def actv(x, args, name):
        if args == 'relu':
            return tf.nn.relu(x, name)

        elif args == 'relu6':
            return tf.nn.relu6(x, name)

        elif args == 'crelu':
            return tf.nn.crelu(x, name)

        elif args == 'elu':
            return tf.nn.elu(x, name)

        elif args == 'selu':
            return tf.nn.selu(x, name)

        elif args == 'softplus':
            return tf.nn.softplus(x, name)

        elif args == 'softsign':
            return tf.nn.softsign(x, name)

        elif args == 'sigmoid':
            return tf.nn.sigmoid(x, name)

        elif args == 'tanh':
            return tf.nn.tanh(x, name)

        else:
            return x

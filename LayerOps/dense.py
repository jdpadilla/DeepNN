import tensorflow as tf
from DeepNN2_0.LayerOps.nnetops import NNetOps

from DeepNN2_0.OtherFncs.initialweights import initial_weights
from DeepNN2_0.OtherFncs.initialbias import initial_bias

from DeepNN2_0.LayerOps.activations import Activations


# Create a Fully Connected Layer
class Dense(NNetOps):

    @staticmethod
    def fc(x, args, name):

        """Create a fully connected layer."""
        iweight = initial_weights(args.iweight, x)
        ibias = initial_bias(args.ibias, x)

        with tf.variable_scope(name) as scope:

            # Create tf variables for the weights and biases
            weights = tf.get_variable(name='weights', initializer=iweight,
                                      shape=[args.num_in, args.num_out], trainable=True)
            biases = tf.get_variable(name='biases', initializer=ibias,
                                     shape=[args.num_out], trainable=True)

            # Matrix multiply weights and inputs and add bias
            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

            result = Activations.actv(act, args.actv, name=scope.name)

            return result

from DeepNN2_0.OtherFncs.initialweights import initial_weights

"""BIAS Initializer"""


def initial_bias(ibias, shape, weights_path=None, session=None):

    initializer = initial_weights(ibias, shape, weights_path, session)

    return initializer
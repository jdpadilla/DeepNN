import abc

# import sys
# sys.path.insert(0, r'/home/jdpadilla/Documents/PyProjects/DeepNN/OtherFncs')
# sys.path.insert(0, r'/home/jdpadilla/Documents/PyProjects/DeepNN/BMModels')
# sys.path.insert(0, r'/home/jdpadilla/Documents/PyProjects/DeepNN/NNetOps')

class NNetOps(object):
    """
    Define all the neural network operations:
    - Dense Layer (Fully Connected)
    - Convolution (Separable, 1D, 2D, 3D, Deconvolution)
    - Normalizations (l2norm, lrn, suffstats, normmoments, moments, wmoments, fbatchnorm, batchnorm, batchnormwgnorm)
    - Pooling (avgp, maxp, argmaxp, avg3d, max3d, fr_avg, fr_max)
    - Activations (relu, relu6, crelu, elu, selu, softplus, softsign, tanh, sigmoid)
    - Dropout
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, x, args, name): #, x, args, name
        # pass
        # """
        # x: placeholder for input tensor on which to perform operations
        # args: arguments for the operation
        # num_classes:
        # keep_prob: for dropout
        # name: label for the op+layer e.g. conv2d-L1
        # """
        self.X = x
        self.ARGS = args
        self.NAME = name
        self.create()

    @abc.abstractmethod
    def create(self):
        pass

    @staticmethod
    def Convolution(x, args, name):
        """
        Create a convolutional layer with
        :param x: the placeholder for the tensor
        :param args: the arguments for the convolution (filter, strides, padding)
        :param name: a label for the operation
        :return: a convolved tensor
        """
        from convolution import Convolution
        return Convolution(x, args, name)

    @staticmethod
    def Dense(x, args, name):
        """
        Create a fully connected layer with
        :param x: the placeholder for the tensor
        :param args: the arguments for the fc layer(num_in, num_out)
        :param name: a label for the operation
        :return: a fully connected layer tensor
        """
        from dense import Dense
        return Dense(x, args, name)

    @staticmethod
    def Dropout(x, args, name):
        """
        Create a dropout layer with
        :param x: the placeholder for the tensor
        :param args: the arguments for the dropout (Keep_prob)
        :param name: a label for the operation
        :return: a "dropped" tensor
        """
        from dropout import Dropout
        return Dropout(x, args, name)

    @staticmethod
    def Normalization(x, args, name):
        """
        Applies normalization to a layer with
        :param x: the placeholder for the tensor
        :param args: the arguments for the normalization (dependent on the kind of normalization)
        :param name: a label for the operation
        :return: a normalized layer tensor
        """
        from normalizations import Normalization
        return Normalization(x, args, name)

    @staticmethod
    def Pooling(x, args, name):
        return Pooling(x, args, name)
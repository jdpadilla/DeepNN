import tensorflow as tf

from DeepNN.OtherFncs.initialbias import initialbias
from DeepNN.OtherFncs.initialweights import initialweights


class Model(object):
    """
    Super Class for all the Architectures to be built
    initialweight(): generate the initial values for the weights
    initialweight(): generate the initial values for the bias
    train(): train the model
    validate(): validate the model
    """
    def __init__(self, x, args):
        """
        x, num_classes, args.dropout_rate, args.train_layers,
             args.iweight, args.ibias, args.model
        :param args.x: placeholde for tensor
        :param args.num_classes: number of classes in the dataset
        :param args.dropout_rate: probability for dropout layer
        :param args.train_layers: layers to be trained from scratch
        :param args.iweight: weight initialization fn
        :param args.ibias: bias initialization fn
        :param args.model: which model to build
        """
        self.x = x
        self.args = args

    # @staticmethod
    def train(self, session):
        pass

    # @staticmethod
    def validate(self, session):
        pass

import argparse
import os

"""
Argument parser for Deep Neural Network
Set all parameters from here
"""


def arg_inputs():

    parser = argparse.ArgumentParser(description='< < Argument Parsing for Deep Neural Network > > >')

    parser.add_argument('-tbd', '--tboard',
                        help='Enable TensorBoard (y/n) [y]',
                        default='y', type=str)

    parser.add_argument('-gpu', '--gpu',
                        help='Enable all GPU\'s available (y/n) [y]',
                        default='n', type=str)

    cwd = os.path.realpath(os.getcwd())

    sfw_dir = os.path.join(cwd, 'summary')
    parser.add_argument('-sfwd', '--sfw_dir',
                        help='SUMMARY FILE WRITER folder [/path/to/DeepNN2.0/summary]',
                        default=sfw_dir, type=str)

    cp_dir = os.path.join(cwd, 'checkpoints')
    os.path.join(cwd, 'checkpoints')
    parser.add_argument('-cpd', '--cp_dir',
                        help='CHECKPOINT folder [/path/to/DeepNN2.0/checkpoints]',
                        default=cp_dir, type=str)

    parser.add_argument('-disp', '--display_step',
                        help='Display Step [20]',
                        default=1, type=int)

    # Data Set, and txt files containing labels /path/to/DeepNN2_0/DataSet/

    datad = os.path.join(cwd, 'DataSet/Imgs')

    parser.add_argument('-datad', '--ds_dir',
                        help='DATASET folder [/path/to/DeepNN2_0/DataSet/Imgs]',
                        default=datad, type=str)

    train_f = os.path.join(cwd, 'DataSet/train.txt')

    parser.add_argument('-trfd', '--train_f',
                        help='Training txt File [path/to/DeepNN2_0/DataSet/]',
                        default=train_f, type=str)

    val_f = os.path.join(cwd, 'DataSet/val.txt')

    parser.add_argument('-vlfd', '--val_f',
                        help='Validation txt File [/path/to/DeepNN2_0/DataSet/]',
                        default=val_f, type=str)

    test_f = os.path.join(cwd, 'DataSet/test.txt')

    parser.add_argument('-tsfd', '--test_f',
                        help='Test txt File [/path/to/DeepNN2_0/DataSet/]',
                        default=test_f, type=str)

    # Learning Parameters
    parser.add_argument('-lr', '--learning_rate',
                        help='Learning Rate [0.0001]',
                        default=0.001, type=float)

    parser.add_argument('-nepochs', '--num_epochs',
                        help='Number of Epochs [10]',
                        default=10, type=int)

    parser.add_argument('-bsize', '--batch_size',
                        help='Batch Size [128]',
                        default=128, type=int)

    # Network parameters
    parser.add_argument('-drop', '--dropout_rate',
                        help='Dropout Rate [0.5]',
                        default=0.5, type=float)

    parser.add_argument('-nclass', '--num_classes',
                        help='Number of Classes [2 (0=Unknown)]',
                        default=2, type=int)

    # train_layers = ['L8', 'L7', 'L6', 'L5', 'L4', 'L3', 'L2', 'L1']
    train_layers = ['L8', 'L7']
    parser.add_argument('-trlayers', '--train_layers',
                        help='Layers to Train [All (Lx, Ly, Lz, ..., Ln)]',
                        default=train_layers, type=list)

    # Model to Build
    parser.add_argument('-model', '--model',
                        help='Model to Build: [alexnet], googlenet, resnet, squeezenet, vgg, dcgan, custom',
                        default='alexnet', type=str)

    # Weight Initializer
    parser.add_argument('-iweight', '--iweight',
                        help='Weight Initialization: constant, [runiform], rnormal, xavier, tnormal, load',
                        default='tnormal', type=str)

    # Bias Initializer
    parser.add_argument('-ibias', '--ibias',
                        help='Bias Initialization: constant, [runiform], rnormal, xavier, tnormal, load',
                        default='tnormal', type=str)

    # Loss Function
    parser.add_argument('-loss', '--lossfn',
                        help='Loss Function: sigmoid_xewl, softmax, [softmax_xewl], sparse_softmax_xewl'
                        'weighted_xewl, Log_softmax',
                        default='softmax_xewl', type=str)

    # Optimizer
    parser.add_argument('-optm', '--optimizer',
                        help='Optimizer Function: rmsprop, proximaladagrad, adagradda, ftrl, adam, proximalgd'
                        '[gd], adadelta, adagrad, momentum',
                        default='gd', type=str)

    return parser.parse_args()

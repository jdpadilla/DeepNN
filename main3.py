"""
Inspired on: Script to finetune AlexNet using Tensorflow by Frederik Kratzert
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
"""
import tensorflow as tf
import os
import sys

sys.path.append('../')

# Helper Files
from DeepNN2_0.OtherFncs.arginputs import arg_inputs
from DeepNN2_0.OtherFncs.updatedirs import update_paths
import numpy as np
from datetime import datetime
from DeepNN2_0.OtherFncs.buildthis import build_this
from DeepNN2_0.Models.alexnet import AlexNet
from DeepNN2_0.OtherFncs.execute import execute

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Parse Arguments
"""
--tboard, Enable TensorBoard
--gpu, Enable all GPU\'s available (y/n) [y]
--sfw_dir, SUMMARY FILE WRITER folder [/path/to/DeepNN2_0/summary]
--cp_dir, CHECKPOINT folder [/path/to/DeepNN2_0/checkpoints]
--display_step, Display Step [20]

--ds_dir, DATASET folder [/path/to/DeepNN2_0/DataSet/Imgs]
--train_f, Training txt File [path/to/DeepNN2_0/DataSet/train.txt]
--val_f, Validation txt File [/path/to/DeepNN2_0/DataSet/val.txt]

--learning_rate, Learning Rate [0.0001]
--num_epochs, Number of Epochs [10]
--batch_size, Batch Size [128]

--dropout_rate, Dropout Rate [0.5]
--num_classes, Number of Classes [3]
--train_layers, Layers to Train ['L8', 'L7', 'L6', 'L5', 'L4', 'L3', 'L2', 'L1']

--model, Model to Build [alexnet]

--iweight, Weight Initialization [xavier]
--ibias, Bias Initialization [xavier]
--lossfn, Loss Function [softmax_xewl]
--optimizer, Optimizer Function [gd]
"""

args = arg_inputs()
print('Running with parameters:', args)

if args.gpu == 'n':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# /path/to/pre-trained/weights
if args.iweight or args.ibias == 'load':
    args.wpath = '/home/jdpadilla/Documents/PyProjects/finetune_alexnet_with_tensorflow/bvlc_alexnet.npy'

# Set the paths and labels for images in the train, val, and test files
update_paths(args)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [args.batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, args.num_classes])
keep_prob = tf.placeholder(tf.float32)

# model = AlexNet(x, args)
model = build_this(x, args)

# Link variable to model output
score = model.fc8

execute(x, y, keep_prob, args, score)

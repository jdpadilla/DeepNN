import tensorflow as tf
import numpy as np

from datetime import datetime

from argparse import Namespace
from DeepNN2_0.OtherFncs.summarize import summarize
from DeepNN2_0.OtherFncs.training import train_model
from DeepNN2_0.OtherFncs.validation import validate_model
from DeepNN2_0.OtherFncs.initialweights import initial_weights
from DeepNN2_0.OtherFncs.initialbias import initial_bias
from DeepNN2_0.OtherFncs.datagenerator import ImageDataGenerator

def execute(x, y, keep_prob, args, score):

    # SUMMARIZE
    merged_summary, train_op, accuracy = summarize(y, args, score)

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(args.sfw_dir)

    # Initialize a saver for storing model checkpoints
    saver = tf.train.Saver()

    # Training and Validation data generator initialized seperately
    train_generator = ImageDataGenerator(args.train_f, training=True)
    val_generator = ImageDataGenerator(args.val_f)

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = int(np.floor(train_generator.data_size / args.batch_size))
    val_batches_per_epoch = int(np.floor(val_generator.data_size / args.batch_size))

    runparams = Namespace(train_batches_per_epoch = train_batches_per_epoch,
                          val_batches_per_epoch = val_batches_per_epoch,
                          train_generator = train_generator,
                          val_generator = val_generator,
                          merged_summary = merged_summary,
                          accuracy=accuracy,
                          train_op = train_op,
                          saver = saver
                          )

    if args.gpu == 'y':
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
    else:
        config = None

    # Start Tensorflow session
    with tf.Session(config=config) as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        if args.iweight == 'load':
            initial_bias(args.iweight, x, args.wpath, sess)
            # initial_weights(args.iweight, x, args.wpath, sess)

        if args.tboard == 'y':
            print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                              args.sfw_dir))

        # Loop over number of epochs
        for epoch in range(args.num_epochs):

            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

            train_model(epoch, sess, writer, x, y, keep_prob,  args, runparams)

            validate_model(epoch, sess, writer, x, y, keep_prob, args, runparams)

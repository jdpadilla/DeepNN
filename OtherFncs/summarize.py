import tensorflow as tf

from DeepNN2_0.OtherFncs.loss import lossfn
from DeepNN2_0.OtherFncs.optimizer import optimizerfn


def summarize(y, args, score):

    # Iterate through all trainable variables and select the ones requested to train from scratch (weight & bias)
    var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0][-2:] in args.train_layers]

    # EVALUATE LOSS
    loss = lossfn(args.lossfn, score, y)

    # OPTIMIZE
    with tf.name_scope("Train"):
        # Get gradients of all trainable variables
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))

        optimizer = optimizerfn(args.optimizer, args.learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients)

    for gradient, var in gradients:
        tf.summary.histogram(var.name + '/Gradient', gradient)

    # Add the variables we train to the summary
    for var in var_list:
        tf.summary.histogram(var.name, var)

    # Add the loss to summary
    tf.summary.scalar('LOSS', loss)

    # ACCURACY
    with tf.name_scope("Accuracy"):
        correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Add the accuracy to the summary
    tf.summary.scalar('ACCURACY', accuracy)

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    return merged_summary, train_op, accuracy

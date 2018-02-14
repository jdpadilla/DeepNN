import tensorflow as tf


def optimizerfn(opt, learning_rate):

    if opt == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate)

    elif opt == 'proximaladagrad':
        optimizer = tf.train.ProximalAdagradOptimizer(learning_rate)

    elif opt == 'adagradda':
        optimizer = tf.train.AdagradDAOptimizer(learning_rate)

    elif opt == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)

    elif opt == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)

    elif opt == 'proximalgd':
        optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate)

    elif opt == 'gd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    elif opt == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)

    elif opt == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate)

    elif opt == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate)

    else:
        optimizer = tf.train.RMSPropOptimizer(learning_rate)

    return optimizer

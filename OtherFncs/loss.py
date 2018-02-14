import tensorflow as tf


def lossfn(opt, score, y):

    if opt == 'sigmoid_xewl':
        with tf.name_scope("Loss_SXEwL"):
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=score,
                                                                          labels=y)
                                  )

    elif opt == 'softmax':
        with tf.name_scope("Loss_SM"):
            loss = tf.reduce_mean(tf.nn.softmax(logits=score))

    elif opt == 'softmax_xewl':
        with tf.name_scope("Loss_SMCEwL"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                          labels=y)
                                  )

    elif opt == 'sparse_softmax_xewl':
        with tf.name_scope("Loss_SSMCEwL"):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=score,
                                                                                 labels=y)
                                  )

    elif opt == 'weighted_xewl':
        with tf.name_scope("Loss_WCEwL"):
            loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=score))

    elif opt == 'Log_softmax':
        with tf.name_scope("Loss_LSM"):
            loss = tf.reduce_mean(tf.nn.log_softmax(logits=score))

    else:
        with tf.name_scope("Loss_SXEwL"):
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=score,
                                                                          labels=y))

    return loss
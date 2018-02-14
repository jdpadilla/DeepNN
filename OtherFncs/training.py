from datetime import datetime
import time

# Model Training Fnc

def train_model(epoch, sess, writer, x, y, keep_prob, args, runparams):
    start = time.perf_counter()
    print("{} Start training".format(datetime.now()))

    for step in range(runparams.train_batches_per_epoch):
        print('Training... Batch Number: ', step)
        # get next batch of data
        img_batch, label_batch = runparams.train_generator.next_batch(args.batch_size)

        # And run the training op
        sess.run(runparams.train_op, feed_dict={x: img_batch,
                                                y: label_batch,
                                                keep_prob: args.dropout_rate})

        # Generate summary with the current batch of data and write to file
        if step % args.display_step == 0:
            s = sess.run(runparams.merged_summary, feed_dict={x: img_batch,
                                                              y: label_batch,
                                                              keep_prob: 1.})

            writer.add_summary(s, epoch * runparams.train_batches_per_epoch + step)

    elapsed = time.perf_counter() - start
    print('Elapsed Training Time: %.3f seconds.' % elapsed)
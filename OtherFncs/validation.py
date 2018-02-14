import os, time
from datetime import datetime

# Model Training Fnc
def validate_model(epoch, sess, writer, x, y, keep_prob, args, runparams):
    start = time.perf_counter()
    print("{} Start validation".format(datetime.now()))

    test_acc = 0.
    test_count = 0

    for step in range(runparams.val_batches_per_epoch):
        print('Validating... Batch Number: ', step)
        img_batch, label_batch = runparams.val_generator.next_batch(args.batch_size)
        acc = sess.run(runparams.accuracy, feed_dict={x: img_batch,
                                                      y: label_batch,
                                                      keep_prob: 1.})
        test_acc += acc
        test_count += 1

    elapsed = time.perf_counter() - start
    print('Elapsed Validation Time: %.3f seconds.' % elapsed)

    test_acc /= test_count
    print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                   test_acc))

    runparams.val_generator.reset_pointer()
    runparams.train_generator.reset_pointer()

    print("{} Saving checkpoint of model...".format(datetime.now()))

    # save checkpoint of the model
    checkpoint_name = os.path.join(args.cp_dir,
                                   'model_epoch' + str(epoch + 1) + '.ckpt')

    save_path = runparams.saver.save(sess, checkpoint_name)

    print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                   checkpoint_name))
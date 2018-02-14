import tensorflow as tf
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
global fmode, nclass  # Used when parsing


def img_feeder(txt_file, mode, batch_size, num_classes, shuffle=True, buffer_size=1000):

    """
    Args:
    txt_file: Path to the text file.
    mode: Either 'training' or 'inference'. Data Augmentation if training
    batch_size: Number of images per batch.
    num_classes: Number of classes in the dataset.
    shuffle: Whether or not to shuffle the data in the dataset and the initial file list.
    buffer_size: Number of images used as buffer for TensorFlow shuffling of the dataset.
    """

    global fmode, nclass
    fmode = mode
    nclass = num_classes

    img_paths, labels = read_txt_file(txt_file)

    if num_classes == 0:  # Determine number of classes based on unique items in the list
        nclass = len(set(labels))

    data_size = len(labels)  # number of samples in the dataset

    if shuffle:
        img_paths, labels = shuffle_lists(img_paths, labels, data_size)

    # convert lists to TF tensor
    img_paths = convert_to_tensor(img_paths, dtype=dtypes.string)
    labels = convert_to_tensor(labels, dtype=dtypes.int32)

    # create dataset
    data = tf.data.Dataset.from_tensor_slices((img_paths, labels))

    # One Hot, Center, and BGR + data augmentation if mode == 'training'
    data = data.map(parse_function,
                    num_parallel_calls=8).prefetch(100 * batch_size)

    # shuffle the first 'buffer_size' elements of the dataset
    if shuffle:
        data = data.shuffle(buffer_size=buffer_size)

    # create a new dataset with batches of images
    data = data.batch(batch_size)

    return data, nclass, data_size


# Read the content of the text file and store it into lists
def read_txt_file(txt_file):

    img_paths = []
    labels = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split('\t')
            img_paths.append(items[0])  # image path
            labels.append(int(items[-1]))  # image label

    return img_paths, labels


# Shuffle list of paths and labels
def shuffle_lists(img_paths, img_labels, data_size):

    path = img_paths
    labels = img_labels
    permutation = np.random.permutation(data_size)
    img_paths = []
    img_labels = []
    for i in permutation:
        img_paths.append(path[i])
        img_labels.append(labels[i])

    return img_paths, img_labels


def parse_function(imgs, label):

    # Label -> One Hot
    one_hot = tf.one_hot(label, nclass)

    img_string = tf.read_file(imgs)
    img_decoded = tf.image.decode_png(img_string, channels=3)
    img_resized = tf.image.resize_images(img_decoded, [227, 227])
    if fmode == 'training':
        data_augment()

    img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

    # img_bgr = img_centered[:, :, ::-1]  # RGB -> BGR

    return img_centered, one_hot  # ,img_bgr


def data_augment():
    pass
    # rotate
    # sharpen
    # blur
    # brighten
    # contrast
    # color


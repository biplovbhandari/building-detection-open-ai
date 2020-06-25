# -*- coding: utf-8 -*-

from functools import partial
import tensorflow as tf


@tf.function
def parse_tfrecord(example_proto, keys_dict=None):
    return tf.io.parse_single_example(example_proto, keys_dict)


@tf.function
def to_tuple(inputs, keys=None, n_features=None):
    inputs_list = [inputs.get(key) for key in keys]
    stacked = tf.stack(inputs_list, axis=0)
    # Convert from CHW to HWC
    stacked = tf.transpose(stacked, [1, 2, 0])
    features = stacked[:, :, :n_features]
    labels = stacked[:, :, n_features:]
    return features, labels


@tf.function
def random_transform(dataset, label):
    x = tf.random.uniform(())

    if x < 0.10:
        dataset, label = tf.image.flip_left_right(dataset), tf.image.flip_left_right(label)
    elif tf.math.logical_and(x >= 0.10, x < 0.20):
        dataset, label = tf.image.flip_up_down(dataset), tf.image.flip_up_down(label)
    elif tf.math.logical_and(x >= 0.20, x < 0.30):
        dataset, label = tf.image.flip_left_right(tf.image.flip_up_down(dataset)),\
                         tf.image.flip_left_right(tf.image.flip_up_down(label))
    elif tf.math.logical_and(x >= 0.30, x < 0.40):
        dataset, label = tf.image.rot90(dataset, k=1), tf.image.rot90(label, k=1)
    elif tf.math.logical_and(x >= 0.40, x < 0.50):
        dataset, label = tf.image.rot90(dataset, k=2), tf.image.rot90(label, k=2)
    elif tf.math.logical_and(x >= 0.50, x < 0.60):
        dataset, label = tf.image.rot90(dataset, k=3), tf.image.rot90(label, k=3)
    elif tf.math.logical_and(x >= 0.60, x < 0.70):
        dataset, label = tf.image.flip_left_right(tf.image.rot90(dataset, k=2)),\
                         tf.image.flip_left_right(tf.image.rot90(label, k=2))
    else:
        dataset, label = dataset, label
    return dataset, label


def get_dataset(files, features, label, patch_shape, training=False, **kwargs):

    keys = features + [label]
    columns = [tf.io.FixedLenFeature(shape=patch_shape, dtype=tf.float32) for k in keys]
    keys_dict = dict(zip(keys, columns))

    parser = partial(parse_tfrecord, keys_dict=keys_dict)

    split_data = partial(to_tuple, keys=keys, n_features=len(features))

    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    dataset = dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # repeat is performed on the calling function
    if training:
        dataset = dataset.map(split_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(random_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset \
            .map(split_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset

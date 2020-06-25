# -*- coding: utf-8 -*-

import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import backend as K


def conv_block(input_tensor, num_filters, dropout_rate, name):
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same', name=f'conv1_{name}')(input_tensor)
    encoder = layers.BatchNormalization(name=f'batchnorm1_{name}')(encoder)
    # encoder = layers.Activation('relu')(encoder)
    encoder = layers.Activation(tf.nn.leaky_relu, name=f'activation1_{name}')(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same', name=f'conv2_{name}')(encoder)
    encoder = layers.BatchNormalization(name=f'batchnorm2_{name}')(encoder)
    # encoder = layers.Activation('relu')(encoder)
    encoder = layers.Activation(tf.nn.leaky_relu, name=f'activation2_{name}')(encoder)
    encoder = layers.SpatialDropout2D(dropout_rate, name=f'spatial_dropout1_{name}')(encoder)
    return encoder


def encoder_block(input_tensor, num_filters, dropout_rate, name):
    encoder = conv_block(input_tensor, num_filters, dropout_rate, name)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name=f'max_pooling_{name}')(encoder)
    return encoder_pool, encoder


def decoder_block(input_tensor, concat_tensor, num_filters, dropout_rate, name):
    decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same', name=f'deconv1_{name}')(
        input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1, name=f'concat1_{name}')
    decoder = layers.BatchNormalization(name=f'batchnorm1_{name}')(decoder)
    # decoder = layers.Activation('relu')(decoder)
    decoder = layers.Activation(tf.nn.leaky_relu, name=f'activation1_{name}')(decoder)

    decoder = layers.Conv2D(num_filters, (3, 3), padding='same', name=f'deconv2_{name}')(decoder)
    decoder = layers.BatchNormalization(name=f'batchnorm2_{name}')(decoder)
    # decoder = layers.Activation('relu')(decoder)
    decoder = layers.Activation(tf.nn.leaky_relu, name=f'activation2_{name}')(decoder)

    decoder = layers.Conv2D(num_filters, (3, 3), padding='same', name=f'deconv3_{name}')(decoder)
    decoder = layers.BatchNormalization(name=f'batchnorm3_{name}')(decoder)
    # decoder = layers.Activation('relu')(decoder)
    decoder = layers.Activation(tf.nn.leaky_relu, name=f'activation3_{name}')(decoder)

    decoder = layers.SpatialDropout2D(dropout_rate, name=f'spatial_dropout1_{name}')(decoder)

    return decoder


def get_model(in_shape, out_classes, dropout_rate=0.2, **kwargs):
    inputs = layers.Input(shape=in_shape, name='input')  # 256
    encoder0_pool, encoder0 = encoder_block(inputs, 32, dropout_rate, 'encoder_block_1')  # 128
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64, dropout_rate, 'encoder_block_2')  # 64
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128, dropout_rate, 'encoder_block_3')  # 32
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256, dropout_rate, 'encoder_block_4')  # 16
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512, dropout_rate, 'encoder_block_5')  # 8

    center = conv_block(encoder4_pool, 1024, dropout_rate, 'center_block')  # center

    decoder4 = decoder_block(center, encoder4, 512, dropout_rate, 'decoder_block_5')  # 16
    decoder3 = decoder_block(decoder4, encoder3, 256, dropout_rate, 'decoder_block_4')  # 32
    decoder2 = decoder_block(decoder3, encoder2, 128, dropout_rate, 'decoder_block_3')  # 64
    decoder1 = decoder_block(decoder2, encoder1, 64, dropout_rate, 'decoder_block_2')  # 128
    decoder0 = decoder_block(decoder1, encoder0, 32, dropout_rate, 'decoder_block_1')  # 256

    outputs = layers.Conv2D(out_classes, (1, 1), activation='sigmoid', name='final_out')(decoder0)

    model = models.Model(inputs=[inputs], outputs=[outputs], name='unet')

    return model


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    true_sum = K.sum(K.square(y_true), -1)
    pred_sum = K.sum(K.square(y_pred), -1)
    return 1 - ((2. * intersection + smooth) / (true_sum + pred_sum + smooth))


def bce_dice_loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred, label_smoothing=0.2) + dice_loss(y_true, y_pred)


def bce_loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred, label_smoothing=0.2)


def build(*args, optimizer=None, loss=None, metrics=None, distributed_strategy=None, **kwargs):
    learning_rate = kwargs.get('learning_rate', 0.001)
    if optimizer == 'sgd_momentum':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    if loss is None:
        loss = keras.losses.BinaryCrossentropy(from_logits=True)

    if metrics is None:
        metrics = [
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            dice_coef,
            f1_m,
        ]

    if distributed_strategy is not None:
        with distributed_strategy.scope():
            model = get_model(*args, **kwargs)
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    else:
        model = get_model(*args, **kwargs)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

# -*- coding: utf-8 -*-

from dotenv import load_dotenv

load_dotenv('.env')

import ast
import datetime
import glob
import os
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import callbacks

from model import dataio, model

# specify directory as data io info
BASEDIR = Path(os.getenv('BASEDIR'))
TRAINING_DIR = BASEDIR / 'training_patches'
TESTING_DIR = BASEDIR / 'testing_patches'
VALIDATION_DIR = BASEDIR / 'validation_patches'
OUTPUT_DIR = BASEDIR / 'output'
MODEL_NAME = os.getenv('MODEL_NAME')
MODEL_CHECKPOINT_NAME = os.getenv('MODEL_CHECKPOINT_NAME')

today = datetime.date.today().strftime('%Y_%m_%d')
iterator = 1
while True:
    model_dir_name = f'{today}_V{iterator}'
    MODEL_SAVE_DIR = OUTPUT_DIR / model_dir_name
    try:
        os.mkdir(MODEL_SAVE_DIR)
    except FileExistsError:
        print(f'> {MODEL_SAVE_DIR} exists, creating another version...')
        iterator += 1
        continue
    break

print('***************************************************************************')

# specify some data structure
FEATURES = ast.literal_eval(os.getenv('FEATURES'))
LABELS = ast.literal_eval(os.getenv('LABELS'))

# patch size for training
PATCH_SHAPE = ast.literal_eval(os.getenv('PATCH_SHAPE'))

# Sizes of the training and evaluation datasets.
TRAIN_SIZE = int(os.getenv('TRAIN_SIZE'))
TEST_SIZE = int(os.getenv('TEST_SIZE'))
VAL_SIZE = int(os.getenv('VAL_SIZE'))

# Specify model training parameters.
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
EPOCHS = int(os.getenv('EPOCHS'))
BUFFER_SIZE = int(os.getenv('BUFFER_SIZE'))

# Rates
USE_ADJUSTED_LR = bool(os.getenv('USE_ADJUSTED_LR'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE'))
MAX_LR = float(os.getenv('MAX_LR'))
MID_LR = float(os.getenv('MID_LR'))
MIN_LR = float(os.getenv('MIN_LR'))
RAMPUP_EPOCHS = int(os.getenv('RAMPUP_EPOCHS'))
SUSTAIN_EPOCHS = int(os.getenv('SUSTAIN_EPOCHS'))

print(f'USE_ADJUSTED_LR: {USE_ADJUSTED_LR}')
print(f'MAX_LR: {MAX_LR}')
print(f'MID_LR: {MID_LR}')
print(f'MIN_LR: {MIN_LR}')
print(f'RAMPUP_EPOCHS: {RAMPUP_EPOCHS}')
print(f'SUSTAIN_EPOCHS: {SUSTAIN_EPOCHS}')

DROPOUT_RATE = float(os.getenv('DROPOUT_RATE'))

# other params
CALLBACK_PARAMETER = os.getenv('CALLBACK_PARAMETER')
LOSS = model.dice_loss

training_files = glob.glob(str(TRAINING_DIR) + '/*')
testing_files = glob.glob(str(TRAINING_DIR) + '/*')
validation_files = glob.glob(str(TRAINING_DIR) + '/*')

# save the parameters
with open(f'{str(MODEL_SAVE_DIR)}/parameters.txt', 'w') as f:
    f.write(f'TRAIN_SIZE: {TRAIN_SIZE}\n')
    f.write(f'TEST_SIZE: {TEST_SIZE}\n')
    f.write(f'VAL_SIZE: {VAL_SIZE}\n')
    f.write(f'BATCH_SIZE: {BATCH_SIZE}\n')
    f.write(f'EPOCHS: {EPOCHS}\n')
    f.write(f'BUFFER_SIZE: {BUFFER_SIZE}\n')
    f.write(f'LEARNING_RATE: {LEARNING_RATE}\n')
    if USE_ADJUSTED_LR:
        f.write(f'USE_ADJUSTED_LR: {USE_ADJUSTED_LR}\n')
        f.write(f'MAX_LR: {MAX_LR}\n')
        f.write(f'MID_LR: {MID_LR}\n')
        f.write(f'MIN_LR: {MIN_LR}\n')
        f.write(f'RAMPUP_EPOCHS: {RAMPUP_EPOCHS}\n')
        f.write(f'SUSTAIN_EPOCHS: {SUSTAIN_EPOCHS}\n')
    f.write(f'DROPOUT_RATE: {DROPOUT_RATE}\n')
    f.write(f'FEATURES: {FEATURES}\n')
    f.write(f'LABELS: {LABELS}\n')
    f.write(f'PATCH_SHAPE: {PATCH_SHAPE}\n')
    f.write(f'CALLBACK_PARAMETER: {CALLBACK_PARAMETER}\n')
    f.write(f'MODEL_NAME: {MODEL_NAME}.h5\n')
    f.write(f'MODEL_CHECKPOINT_NAME: {MODEL_CHECKPOINT_NAME}.h5\n')

# get training, testing, and eval TFRecordDataset
# training is batched, shuffled, transformed, and repeated
training = dataio.get_dataset(training_files, FEATURES, LABELS, PATCH_SHAPE, BATCH_SIZE,
                              buffer_size=BUFFER_SIZE, training=True).repeat()
# testing is batched by 1 and repeated
testing = dataio.get_dataset(testing_files, FEATURES, LABELS, PATCH_SHAPE, 1).repeat()
# eval is batched by 1
validation = dataio.get_dataset(validation_files, FEATURES, LABELS, PATCH_SHAPE, 1)

# get distributed strategy and apply distribute i/o and model build
strategy = tf.distribute.MirroredStrategy()

# define tensor input shape and number of classes
# in_shape = (None, None) + (len(FEATURES),)
in_shape = [None, None, len(FEATURES)]
out_classes = int(os.getenv('OUT_CLASSES_NUM'))

# build the model and compile
my_model = model.build(in_shape, out_classes, distributed_strategy=strategy, dropout_rate=DROPOUT_RATE,
                       learning_rate=LEARNING_RATE, loss=LOSS)
print(my_model.summary())

# define callbacks during training
model_checkpoint = callbacks.ModelCheckpoint(
    f'{str(MODEL_SAVE_DIR)}/{MODEL_CHECKPOINT_NAME}.h5',
    monitor=CALLBACK_PARAMETER, save_best_only=True,
    mode='min', verbose=1, save_weights_only=True
)
early_stopping = callbacks.EarlyStopping(
    monitor=CALLBACK_PARAMETER, patience=5, verbose=0,
    mode='auto', restore_best_weights=True
)
tensorboard = callbacks.TensorBoard(log_dir=str(MODEL_SAVE_DIR / 'logs'), write_images=True)


def lr_scheduler(epoch):
    if epoch < RAMPUP_EPOCHS:
        return MAX_LR
    elif epoch < RAMPUP_EPOCHS + SUSTAIN_EPOCHS:
        return MID_LR
    else:
        return MIN_LR


lr_callback = callbacks.LearningRateScheduler(lambda epoch: lr_scheduler(epoch), verbose=True)

model_callbacks = [model_checkpoint, tensorboard, early_stopping]
if USE_ADJUSTED_LR:
    model_callbacks.append(lr_callback)

# fit the model
history = my_model.fit(
    x=training,
    epochs=EPOCHS,
    steps_per_epoch=(TRAIN_SIZE // BATCH_SIZE),
    validation_data=testing,
    validation_steps=TEST_SIZE,
    callbacks=model_callbacks,
)

# check how the model trained
my_model.evaluate(validation)

# save the model
my_model.save(f'{str(MODEL_SAVE_DIR)}/{MODEL_NAME}.h5')

# open and save model
this_model = model.get_model(in_shape, out_classes, dropout_rate=DROPOUT_RATE)
this_model.load_weights(f'{str(MODEL_SAVE_DIR)}/{MODEL_CHECKPOINT_NAME}.h5')
tf.keras.models.save_model(this_model, str(MODEL_SAVE_DIR), save_format='tf')

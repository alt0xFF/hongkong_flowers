from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import glob
import numpy as np
import keras
from keras.applications import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense

np.random.seed(1337)  # for reproducibility

def get_class_weight(d):
    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}
    class_number = dict()
    dirs = sorted([o for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))])
    k = 0
    for class_name in dirs:
        class_number[k] = 0
        iglob_iter = glob.iglob(os.path.join(d, class_name, '*.*'))
        for i in iglob_iter:
            _, ext = os.path.splitext(i)
            if ext[1:] in white_list_formats:
                class_number[k] += 1
        k += 1

    total = np.sum(list(class_number.values()))
    max_samples = np.max(list(class_number.values()))
    mu = 1. / (total / float(max_samples))
    keys = class_number.keys()
    class_weight = dict()
    for key in keys:
        score = math.log(mu * total / float(class_number[key]))
        class_weight[key] = score if score > 1. else 1.

    return class_weight

def get_callbacks(weights_path, patience=30, monitor='val_loss'):
    early_stopping = EarlyStopping(verbose=1, patience=patience, monitor=monitor)
    model_checkpoint = ModelCheckpoint(weights_path, save_best_only=True, save_weights_only=True, monitor=monitor)
    reduce_leanring_rate = ReduceLROnPlateau(monitor="val_acc", verbose=1)
    return [reduce_leanring_rate, early_stopping, model_checkpoint]

# path to the model weights files.
weights_path = 'trained/fine-tuned-resnet50-weights.h5'

# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'data/oxford102/train'
validation_data_dir = 'data/oxford102/valid'
test_data_dir = 'data/oxford102/test'
classes = sorted([o for o in os.listdir(train_data_dir) if os.path.isdir(os.path.join(train_data_dir, o))])
epochs = 1000
batch_size = 32
nb_classes = len(classes)
patience = 20
class_weight = get_class_weight(train_data_dir)

# build the VGG16 network
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
print('Model loaded.')

# set the first 80 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers:
    layer.trainable = False

for layer in model.layers[80:]:
    layer.trainable = True

# build a classifier model to put on top of the convolutional model
y = model.output
y = Flatten()(y)
y = Dropout(0.5)(y)

# now the shape = (batch_size, 2048)
y = Dense(2048, activation='elu', name='fc1')(y)
y = Dropout(0.5)(y)
y = Dense(1024, activation='elu', name='fc2')(y)
y = Dropout(0.5)(y)
y = Dense(512, activation='elu', name='fc3')(y)
y = Dropout(0.5)(y)

predictions = Dense(nb_classes, activation='softmax', name='predictions2')(y)

model = Model(model.input, predictions, name='resnet50')

# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# fine-tune the model
model.fit_generator(
    train_generator,
    callbacks=get_callbacks(weights_path, patience=patience),
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    class_weight=class_weight)

# evalue
result = model.evaluate_generator(
    generator=test_generator,
    steps=test_generator.samples // batch_size)

print("Model [loss, accuracy]: {0}".format(result))

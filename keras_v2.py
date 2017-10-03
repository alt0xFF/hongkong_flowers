import os
import numpy as np
import keras
from keras.applications import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense

np.random.seed(1337) # for reproducibility

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # only use first GPU

def get_callbacks(weights_path, patience=30, monitor='val_loss'):
    early_stopping = EarlyStopping(verbose=1, patience=patience, monitor=monitor)
    model_checkpoint = ModelCheckpoint(weights_path, save_best_only=True, save_weights_only=True, monitor=monitor)
    return [early_stopping, model_checkpoint]

# path to the model weights files.
finetuned_weights_path = 'trained/fine-tuned-resnet50-weights.h5'
weights_path = 'trained/hk_flower.h5'

# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'data/train'
validation_data_dir = 'data/valid_aug'
test_data_dir = 'data/test'
epochs = 1000
batch_size = 32
nb_classes = 168
patience = 20

# build the ResNet50 network
model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
print('Model loaded.')

# only train last res block
for layer in model.layers:
    layer.trainable = False
for layer in model.layers[80:]:
    layer.trainable = True

# build a classifier model to put on top of the convolutional model
y = model.output
y = Flatten()(y)
y = Dropout(0.5)(y)

# now the shape = (batch_size, 4096)
y = Dense(4096, activation='elu', name='my_fc1')(y)
y = Dropout(0.5)(y)

predictions = Dense(nb_classes, activation='softmax', name='predictions')(y)

model = Model(model.input, predictions, name='resnet50')

model.load_weights(finetuned_weights_path, by_name=True)
print('Weight loaded.')

# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
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

model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size)

model.fit_generator(
    train_generator,
    callbacks=get_callbacks(weights_path, patience=patience),
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size)

# evalue
result = model.evaluate_generator(
    generator=test_generator,
    steps=test_generator.samples // batch_size)

print("Model [loss, accuracy]: {0}".format(result))

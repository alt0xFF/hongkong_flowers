import numpy as np
from keras import backend as K
from keras.layers import *

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam

np.random.seed(1337)  # for reproducibility

batch_size = 32
num_epochs = 1000

width = 300
height = 300

# count folders in data_dir
num_classes = 168

# Input
inputs = Input(name='the_input', shape=(300, 300, 3), dtype='float32')

# input shape = (batch_size, 300, 300, 3)
conv1 = Conv2D(64, (3, 3), activation='relu')(inputs)
maxpool1 = MaxPooling2D(pool_size=(4, 4))(conv1)

conv2 = Conv2D(128, (3, 3), activation='relu')(maxpool1)
maxpool2 = MaxPooling2D(pool_size=(4, 4))(conv2)

conv3 = Conv2D(256, (3, 3), activation='relu')(maxpool2)
maxpool3 = MaxPooling2D(pool_size=(4, 4))(conv3)

conv4 = Conv2D(512, (3, 3), activation='relu')(maxpool3)
maxpool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

# now the shape = (batch_size, 1, 1, 512), so need to squeeze twice.
features = Lambda(lambda x: K.squeeze(x, 1))(maxpool4)
features = Lambda(lambda x: K.squeeze(x, 1))(features)

# now the shape = (batch_size, 512)
y_pred = Dense(num_classes, activation='softmax')(features)

# now y_pred has the shape = (batch_size, num_classess)
model = Model(inputs=[inputs], outputs=[y_pred])

# compile model
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=5e-6), metrics=['accuracy'])

# datasets and dataloader for train mode
train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory('./data/train/',
                                                           batch_size=batch_size,
                                                           target_size=(height, width),
                                                           class_mode='categorical')

valid_datagen = ImageDataGenerator()
valid_generator = valid_datagen.flow_from_directory('./data/valid/',
                                                           batch_size=batch_size,
                                                           target_size=(height, width),
                                                           class_mode='categorical')

train_step = train_generator.samples // batch_size
valid_step = valid_generator.samples // batch_size

# # Generate dummy data
# x_train = np.random.random((10, 300, 300, 3))
# y_train = np.random.randint(num_classes, size=(10, 1))

# # for trying out few images only
# from PIL import Image
# x_train = []
# y_train = []
# import os
# train_dir = './data/train/'
# dir_list = os.listdir(train_dir)
# dir_list.remove('.DS_Store')
# for i, dir in enumerate(dir_list):
#     try:
#         x_train.append(np.asarray(Image.open(train_dir + "%s/00.jpg" % dir)))
#     except IOError:
#         x_train.append(np.asarray(Image.open(train_dir + '%s/01.jpg' % dir)))
#     y_train.append(i)
# x_train = np.asarray(x_train)
# y_train = np.asarray(y_train)
#
# model.fit(x_train, y_train,
#           epochs=2000,
#           batch_size=2)

# fit the model
model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_step,
                    epochs=num_epochs,
                    validation_data=valid_generator,
                    validation_steps=valid_step)

from keras import backend as K
from keras.layers import *
from keras import regularizers
from keras.models import Model, Sequential

# note that for keras these are functions!
def model_template(args, num_classes):

    # Input
    inputs = Input(name='the_input', shape=args.img_size, dtype='float32')

    # input shape = (batch_size, 300, 300, 3)
    conv1 = Conv2D(64, (3, 3), activation='relu')(inputs)
    print(conv1.shape)
    maxpool1 = MaxPooling2D(pool_size=(4, 4))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(4, 4))(conv2)
    print(conv2.shape)

    conv3 = Conv2D(256, (3, 3), activation='relu')(maxpool2)
    maxpool3 = MaxPooling2D(pool_size=(4, 4))(conv3)
    print(conv3.shape)

    conv4 = Conv2D(512, (3, 3), activation='relu')(maxpool3)
    maxpool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    print(conv4.shape)

    # now the shape = (batch_size, 1, 1, 512), so need to squeeze twice.
    features = Lambda(lambda x: K.squeeze(x, 1))(maxpool4)
    features = Lambda(lambda x: K.squeeze(x, 1))(features)
    print(features.shape)

    # now the shape = (batch_size, 512)
    y_pred = Dense(num_classes, activation='softmax')(features)
    print(y_pred.shape)

    # now y_pred has the shape = (batch_size, num_classess)
    model = Model(inputs=[inputs], outputs=[y_pred])

    return model

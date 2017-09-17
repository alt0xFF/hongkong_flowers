from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.layers import *
from tensorflow.contrib.keras.python.keras import regularizers
from tensorflow.contrib.keras.python.keras.models import Model, Sequential

# note that for keras these are functions!
def model_template(args, inputs, num_classes):
    # input shape = (batch_size, 300, 300, 3)
    conv1 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(inputs)
    maxpool1 = MaxPooling2D(pool_size=(4, 4))(conv1)
    
    conv2 = Conv2D(128, (3, 3), strides=(1, 1), activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(4, 4))(conv2)
    
    conv3 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu')(maxpool2)
    maxpool3 = MaxPooling2D(pool_size=(4, 4))(conv3)    
    
    conv4 = Conv2D(512, (3, 3), strides=(1, 1), activation='relu')(maxpool3)
    maxpool4 = MaxPooling2D(pool_size=(2, 2))(conv4)    
    
    # now the shape = (batch_size, 1, 1, 512), so need to squeeze twice.
    features = Lambda(lambda x: K.squeeze(x, 1))(maxpool4)
    features = Lambda(lambda x: K.squeeze(x, 1))(features)
    
    # now the shape = (batch_size, 512)
    dense = Dense(num_classes)(features)
    y_pred = Activation('softmax', name='softmax')(dense)
    
    model = Model(inputs=inputs, outputs=y_pred)

    return model

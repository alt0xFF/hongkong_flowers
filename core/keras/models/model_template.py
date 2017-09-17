from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.layers import *
from tensorflow.contrib.keras.python.keras import regularizers
from tensorflow.contrib.keras.python.keras.models import Model, Sequential

# note that for keras these are functions!
def model_template(inputs, num_classes):
    conv1 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(inputs)
    maxpool1 = MaxPooling2D(pool_size=(4, 4))(conv1)
    
    conv2 = Conv2D(128, (3, 3), strides=(1, 1), activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(4, 4))(conv2)
    
    conv3 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu')(maxpool2)
    maxpool3 = MaxPooling2D(pool_size=(4, 4))(conv3)    
    
    conv4 = Conv2D(512, (3, 3), strides=(1, 1), activation='relu')(maxpool3)
    maxpool4 = MaxPooling2D(pool_size=(4, 4))(conv4)    
    
    features = Lambda(lambda x: K.squeeze(x))(maxpool4)
    
    dense = Dense(num_classes)(features)
    y_pred = Activation('softmax', name='softmax')(dense)
    
    model = Model(inputs=inputs, outputs=y_pred)

    return model

from keras import backend as K
from keras.layers import Flatten, Dense, Dropout
from keras import regularizers
from keras.models import Model, Sequential, load_model
from keras.applications import ResNet50 as KerasResNet50

def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))

# note that for keras these are functions!
def ResNet50(args, num_classes):
    base_model = KerasResNet50(include_top=False, input_shape=args.input_shape)

    if args.weights_path:
        base_model.load_weights(filepath=args.weights_path, by_name=True)
        # make layers non trainable
        for layer in base_model.layers:
            layer.trainable = False

    x = base_model.output
    print(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)

    # now the shape = (batch_size, 2048)
    x = Dense(2048, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(base_model.input, predictions)

    return model

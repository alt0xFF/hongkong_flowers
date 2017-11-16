from keras.layers import *
from keras.models import Model, Sequential
from keras.applications.resnet50 import ResNet50, preprocess_input
import os
# note that for keras these are functions!
def resnet50_model(args, num_classes, input_tensor=None):

    # build the ResNet50 network
    model = ResNet50(weights='imagenet', include_top=False, input_shape=args.img_size, input_tensor=input_tensor)
    print('Model loaded.')

    # freezed all layers of resnet50
    # since the weights are pretrained finetuned with oxford102
    # we only train the last fc layer
    for layer in model.layers:
        layer.trainable = True

    # build a classifier model to put on top of the convolutional model
    y = model.output
    y = Flatten()(y)
    y = Dropout(args.dropout)(y)

    # now the shape = (batch_size, 2048)
    y = Dense(2048, activation='elu', name='new_fc1')(y)
    y = Dropout(args.dropout)(y)

    predictions = Dense(num_classes, activation='softmax', name='new_predictions2')(y)

    model = Model(model.input, predictions, name='resnet50')

    # load pretrained model
    assert os.path.isfile(args.pretrained_file)

    try:
        model.load_weights(args.pretrained_file, by_name=True)
        print("Successfully loaded pretrained weights from %s" % args.pretrained_file)
    except:
        raise ValueError('Cannot find any model weights file in %s!' % args.pretrained_file)

    return model

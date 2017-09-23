import os

# custom modules
from options import Options

def train():
    # setting up, options contains all our params
    options = Options(library=0,    # use keras
                      configs=2,    # use resnet50 model
                      transform=1)  # use transform for resnet50

    # set options to your specific experiment
    options.experiment = "fine_tuned_oxford102_model"
    options.gpu = 2
    options.lr = 1E-3
    options.batch_size = 128

    # load the weight file
    options.load = True
    options.load_dir = '/workspace/hongkong_flowers/pretrained_models/fine-tuned-resnet50-weights.h5'

    # choose model from library
    if options.library == 'pytorch':
        from core.pytorch.model import FlowerClassificationModel
    elif options.library == 'keras':
        from core.keras.model import FlowerClassificationModel
    else:
        raise ValueError('Library not supported.')

    # initialize model
    model = FlowerClassificationModel(options)

    # fit model
    model.fit()


if __name__=='__main__':
    train()

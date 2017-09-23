import os

# custom modules
from options import Options

def train():
    # setting up, options contains all our params
    options = Options()

    # set which gpu to use
    # if options.gpu is not None:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "%i" % (3 - options.gpu)

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

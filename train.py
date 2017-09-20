import os

# custom modules
from options import Options

def train():
    # setting up, options contains all our params
    options = Options()

    # if options.gpu is not None:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "%i" % (3 - options.gpu)

    # choose model from library
    if options.library == 'pytorch':
        from core.pytorch.model import Model
        model = Model(options)
    elif options.library == 'keras':
        from core.keras.model import FlowerModel
        model = FlowerModel(options)

    # fit model
    model.fit()


if __name__=='__main__':
    train()

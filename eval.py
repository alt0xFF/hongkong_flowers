# custom modules
from options import Options

def eval():
    # setting up, options contains all our params
    options = Options(library=0,    # use keras
                      configs=2,    # use resnet50 model
                      transform=1)  # use transform for resnet50

    # set options to the specific experiment you are testing on
    # options.gpu = 1
    options.test_batch_size = 84  # since there are 336 test examples

    # load the weight file
    options.load = True
    options.load_file = './checkpoints/2017-10-02_experiment_0/model_best_weights.h5'

    # initialize model
    model = options.initializeModel()

    # fit model
    model.evaluate()


if __name__=='__main__':
    eval()

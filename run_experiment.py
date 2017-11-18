# custom modules
from options import Options


def train():
    # setting up, options contains all our params
    options = Options(library=0,    # use keras
                      configs=2,    # use resnet50 model
                      transform=1)  # use transform for resnet50

    # set options to your specific experiment
    options.experiment = "fine_tuned_oxford102_model_dropout"
    options.dropout = 0.1
    options.number = options.dropout

    # settings
    options.gpu = 2
    options.save_test_result = True

    # early stopping
    options.early_stopping = True
    options.patience = 20

    # general hyperparameters
    options.lr = 1E-2
    options.batch_size = 128

    # reduce lr on plateau
    options.reduce_lr = 0.5

    for i in range(0, 9):

        # initialize model
        model = options.initializeModel()

        # fit model
        model.fit()

        # evaluate model
        model.evaluate()

        # reset model for next parameter
        model.reset()

        # change dropout from 0.1 to 0.9
        options.dropout += 0.1

        # change the log number saved to checkpoints
        options.number = options.dropout


if __name__=='__main__':
    train()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime

LIBRARIES = ['keras',    # 0
             'pytorch',  # 1
             ]

CONFIGS = [
#      model name,   optimizer,            loss              metric
[ "model_template",     "adam",        "CE_loss",    "simple_metric"],  # 0
[ "model_template",      "sgd",        "CE_loss",    "simple_metric"],  # 1
[ "resnet50_model",      "sgd",        "CE_loss",    "simple_metric"],  # 2
]
TRANSFORMS = [
    'to_tensor_only',      #0
    'to_resnet50_format',  #1
             ]


class Options(object):   # NOTE: shared across all modules
    def __init__(self, library=1, configs=0, transform=0):

        # TODO: NOT YET SET UP LOGGER
        self.verbose     = 2            # 0(warning) | 1(info) | 2(debug)

        # training signature for logging
        self.experiment  = "experiment"      # "experiment"
        self.number         = 0              # the current experiment number]
        self.timestamp      = str(datetime.datetime.today().strftime('%Y-%m-%d'))   # "yymmdd"

        # training configuration
        self.library          = library            # choose from LIBRARIES
        self.configs          = configs            # choose from CONFIGS
        self.transform        = transform          # choose from TRANSFORMS
        self.gpu              = 0                  # choose which GPU to use if any ( None for CPU)

        # general hyperparameters
        self.num_epochs       = 1000
        self.lr               = 1E-4
        self.batch_size       = 32
        self.valid_batch_size = 32
        self.test_batch_size  = 64
        self.grad_clip_norm   = 100000

        # regularizers hyperparameters
        self.dropout          = 0.5
        self.l2               = 0.01

        # set early stopping and logging toggles
        self.early_stopping   = False        # Toggle early_stopping
        self.patience         = 10           # number of epochs to consider before early stopping
        self.log_interval     = 1            # print every log_interval batch
        self.visualize        = False        # tensorboard for keras, visdom for pytorch

        # saving models
        self.save_best        = True         # saves the best model weights
        self.save_latest      = True         # saves the latest model weights
        self.log_history      = True         # logs down all the epoch results

        # loading models when training
        self.load             = False        # load previous model weights when training if it exists
        self.load_best        = False         # load best model weights if true, latest if falses
        self.load_dir = "/workspace/hongkong_flowers/pretrained_model/"  # only need this when load=True

        # USE YOUR OWN DATA DIR: NEED ABSOLUTE PATH!
        self.data_dir         = '/workspace/hongkong_flowers/dataset/sorted/'
        self.log_dir          = "/workspace/hongkong_flowers/logs/"
        self.ckpt_dir         = '/workspace/hongkong_flowers/checkpoints/'

        # for the image size
        self.width     = 300
        self.height    = 300
        self.channel   = 3

        self.library   = LIBRARIES[self.library]
        self.configs   = CONFIGS[self.configs]
        self.transform = TRANSFORMS[self.transform]
#----------------------------------------------------------------------------------------#
# advance settings

        # pytorch settings
        if self.library == "pytorch":
            self.use_cuda           = True
            self.dataparallel       = False

        # keras settings
        if self.library == "keras":
            self.img_size  = (self.height, self.width, self.channel)


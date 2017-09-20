from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

LIBRARIES = ['pytorch',  # 0
             'keras']    # 1

CONFIGS = [
#      model name,   optimizer,            loss              metric
[ "model_template",     "adam",        "CE_loss",    "simple_metric"],  # 0
[ "model_template",      "sgd",        "CE_loss",    "simple_metric"],  # 1
[ "model_template",      "sgd",       "NLL_loss",    "simple_metric"],  # 2
]
TRANSFORMS = [
    'to_tensor_only', #0
             ]


class Options(object):   # NOTE: shared across all modules
    def __init__(self):

        # TODO: NOT YET SET UP LOGGER
        self.verbose     = 2            # 0(warning) | 1(info) | 2(debug)

        # training signature for logging
        self.machine     = "machine_id"    # "machine_id"
        self.timestamp   = "17091700"   # "yymmdd##"
        self.gpu         = 0
        # training configuration
        self.library          = 1            # choose from LIBRARIES
        self.configs          = 0            # choose from CONFIGS
        self.transform        = 0            # choose from TRANSFORMS
        self.num_epochs       = 1000
        self.lr               = 0.001
        self.batch_size       = 32
        self.valid_batch_size = 32
        self.test_batch_size  = 32
        self.grad_clip_norm   = 100000

        self.log_interval     = 1            # print every log_interval batch
        self.visualize        = True         # tensorboard for keras, visdom for pytorch
        self.save_best        = False        # save model w/ highest reward if True, otherwise always save the latest model

        # USE YOUR OWN DATA DIR: NEED ABSOLUTE PATH!
        self.data_dir         = '/workspace/hongkong_flowers/dataset/sorted/'

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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import visdom
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

LIBRARIES = ['pytorch', # 0
             'keras']   # 1

MODELS = ['model_template', # 0
         ]            # 1

CONFIGS = [
#      model name,   optimizer,         loss 
[ "model_template",     "adam",     "CE_loss"],  # 0
[ "model_template",      "sgd",     "CE_loss"],  # 1    
]
TRANSFORMS = ['to_tensor_only', #0
             ]

class Options(object):   # NOTE: shared across all modules
    def __init__(self):
        
        #TODO: NOT YET SET UP LOGGER 
        self.verbose     = 0            # 0(warning) | 1(info) | 2(debug)        

        # training signature for logging
        self.machine     = "machine_id"    # "machine_id"
        self.timestamp   = "17091700"   # "yymmdd##"
        
        # training configuration 
        self.library          = 0            # choose from LIBRARIES
        self.configs          = 0            # choose from CONFIGS
        self.transform        = 0            # choose from TRANSFORMS
        self.num_epochs       = 1000
        self.lr               = 0.001
        self.batch_size       = 32
        self.valid_batch_size = 32
        self.test_batch_size  = 32
        self.grad_clip_norm   = 100000
        
        self.log_interval     = 1            # print every log_interval batch
        self.visualize        = True         # whether do online plotting and stuff or not
        self.save_best        = False        # save model w/ highest reward if True, otherwise always save the latest model
        
        # USE YOUR OWN DATA DIR: NEED ABSOLUTE PATH!
        self.data_dir         = '/workspace/hongkong_flowers/dataset/'
        self.mode             = None         # no need to set here: 0(train) | 1(test)

#----------------------------------------------------------------------------------------#
# advance settings

        self.library   = LIBRARIES[self.library]
        self.configs   = CONFIGS[self.configs]
        self.transform = TRANSFORMS[self.transform]
        # pytorch settings
        if self.library == "pytorch":
            self.use_cuda           = torch.cuda.is_available()
            self.dataparallel       = False
            self.dtype              = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
            
            # TODO: set up visdom
            if self.visualize:
                self.vis = visdom.Visdom()

         #TODO: set up keras

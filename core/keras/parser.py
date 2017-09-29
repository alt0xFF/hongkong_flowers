from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# models
from core.keras.models.model_template import model_template
from core.keras.models.resnet50_model import resnet50_model

ModelsDict = {"model_template": model_template,  # contains only CNN and dense layer
              "resnet50_model": resnet50_model,  #
              }
# losses
from keras.losses import categorical_crossentropy

LossesDict = {"CE_loss": categorical_crossentropy,
              }
# optims
from keras.optimizers import SGD, Adam

OptimsDict = {"adam": Adam,
              "sgd": SGD,
              }

# metrics
from keras.metrics import categorical_accuracy

MetricsDict = {"simple_metric": [categorical_accuracy],
               }

# transforms
from keras.preprocessing.image import ImageDataGenerator

TransformsDict = {"to_tensor_only": ImageDataGenerator(rescale=1. / 255),  # note that we need to rescale the image!
                  "to_resnet50_format": ImageDataGenerator(rescale=1. / 255,
                                                           rotation_range=40,
                                                           width_shift_range=0.2,
                                                           height_shift_range=0.2,
                                                           shear_range=0.2,
                                                           zoom_range=0.2,
                                                           horizontal_flip=True,
                                                           fill_mode='nearest'),
                  }

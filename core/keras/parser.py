# models
from core.keras.models.model_template import model_template

ModelsDict = {"model_template": model_template,  # contains only CNN and dense layer
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
                  }

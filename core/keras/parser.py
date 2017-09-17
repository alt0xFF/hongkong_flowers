# models
from core.keras.models.model_template import model_template 
ModelsDict     = {"model_template": model_template,        # contains only CNN and dense layer
                 }
# losses
from tensorflow.contrib.keras.python.keras.losses import categorical_crossentropy
LossesDict     = {"CE_loss": categorical_crossentropy,
                 }
# optims
from tensorflow.contrib.keras.python.keras.optimizers import SGD, Adam
OptimsDict     = {"adam": Adam,
                  "sgd": SGD,
                 }

from tensorflow.contrib.keras.python.keras.metrics import categorical_accuracy
MetricsDict    = {"simple_metric":categorical_accuracy,
                 }

from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator
TransformsDict = {"to_tensor_only": ImageDataGenerator(),
                 }
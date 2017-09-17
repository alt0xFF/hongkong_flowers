# models
from core.keras.models.model_template import model_template 
ModelsDict     = {"model_template": model_template,        # contains only CNN and dense layer
                 }
# losses
from tensorflow.contrib.keras.python.keras.losses import sparse_categorical_crossentropy
LossesDict     = {"CE_loss": sparse_categorical_crossentropy,
                  "NLL_loss":NLLloss,
                 }
# optims
from tensorflow.contrib.keras.python.keras.optimizers import SGD, Adam
OptimsDict     = {"adam": Adam,
                  "sgd": SGD,
                 }

from core.keras.metric import simple_metric
MetricsDict    = {"simple_metric":simple_metric,
                 }

from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator
TransformsDict = {"to_tensor_only": ImageDataGenerat,
                 }
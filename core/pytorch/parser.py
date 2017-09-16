# models
from core.pytorch.models.model_template import model_template 
ModelsDict     = {"model_template": model_template,        # contains only a dense layer
                 }
# losses
from torch.nn import CrossEntropyLoss
LossesDict     = {"CE_loss": CrossEntropyLoss,
                 }
# optims
from torch.optim import Adam, SGD
OptimsDict     = {"adam": Adam,
                  "sgd": SGD,
                 }

from torchvision import transforms
TransformsDict = {"to_tensor_only": transforms.ToTensor(),
                 }
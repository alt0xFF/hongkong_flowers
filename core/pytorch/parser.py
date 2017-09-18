# models
from core.pytorch.models.model_template import model_template 
ModelsDict     = {"model_template": model_template,        # contains only a dense layer
                 }
# losses
from torch.nn import CrossEntropyLoss, NLLLoss
LossesDict     = {"CE_loss": CrossEntropyLoss,
                  "NLL_loss":NLLLoss,
                 }
# optims
from torch.optim import Adam, SGD
OptimsDict     = {"adam": Adam,
                  "sgd": SGD,
                 }

from core.pytorch.metric import simple_metric
MetricsDict    = {"simple_metric":simple_metric,
                 }

from torchvision import transforms
TransformsDict = {"to_tensor_only": transforms.ToTensor(),
                 }
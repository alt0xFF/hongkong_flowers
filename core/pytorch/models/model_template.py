import torch
import torch.nn as nn

class model_template(nn.Module):
    
    def __init__(self, args, num_classes):
        super(model_template, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )


        
    def forward(self, x):
        print(x.size())
        y = self.dense(x)
        print(y.size())
        return y

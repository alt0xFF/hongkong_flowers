import torch
import torch.nn as nn

class model_template(nn.Module):
    
    def __init__(self, args, num_classes):
        super(model_template, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d((4, 4)),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d((4, 4)),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d((4, 4)),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        
        self.dense = nn.Sequential(
            nn.Linear(512, num_classes),
        )
        
    def forward(self, x):
        y = self.conv(x)
        y = torch.squeeze(y)
        z = self.dense(y)
        return z

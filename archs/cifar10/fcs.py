import torch
import torch.nn as nn

class fcs(nn.Module):

    def __init__(self, num_classes=10):
        super(fcs, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(3*32*32, num_classes)
            #nn.ReLU(inplace=True),
            #nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
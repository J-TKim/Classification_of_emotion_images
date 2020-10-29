import torch.nn as nn
from torchvision import models

class pretrained_resnet152(nn.Module):

    def __init__(self):
        
        super(pretrained_resnet152, self).__init__()
        
        self.resnet = models.resnet152(pretrained=True) 
        self.l1 = nn.Linear(1000 , 256)
        self.dropout = nn.Dropout(0.75)
        self.l2 = nn.Linear(256,6)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.resnet(input)
        x = x.view(x.size(0),-1)
        x = self.dropout(self.relu(self.l1(x)))
        x = self.l2(x)
        
        return x
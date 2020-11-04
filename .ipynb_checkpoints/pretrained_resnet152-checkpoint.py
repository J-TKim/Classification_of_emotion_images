import torch.nn as nn
from torchvision import models


class pretrained_resnet152(nn.Module):

    def __init__(self):
        
        super(pretrained_resnet152, self).__init__()
        
        self.resnet = models.resnet152(pretrained=True) 
        self.l1 = nn.Linear(1000 , 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.75)
        self.l2 = nn.Linear(1024, 6)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        

    def forward(self, input):
        x = self.resnet(input)
        x = x.view(x.size(0),-1)
        x = self.l1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.l2(x)
        x = self.LogSoftmax(x)
        
        return x
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn


# In[13]:


import sys
import numpy as np
import matplotlib.pyplot as plt

from mnist_classification.models import resnet_model
from torchvision import datasets, transforms


# In[14]:


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./dataset", train=True, download=True,
                  transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.1307,), (0.3081,))
                  ])),
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./dataset", train=False,
                  transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.1307,), (0.3081,))
                  ])),
    batch_size=test_batch_size,
    shuffle=True)


# In[4]:


model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()


# In[5]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# In[6]:


def load(fn, device):
    d = torch.load(fn, map_location=device)
    
    return d['config'], d['model']


# In[7]:


def plot(x, y_hat):
    for i in range(x.size(0)):
        img = (np.array(x[i].detach().cpu(), dtype='float')).reshape(28,28)

        plt.imshow(img, cmap='gray')
        plt.show()
        print("Predict:", float(torch.argmax(y_hat[i], dim=-1)))


# In[8]:


# Load MNIST test set.
x, y = load_mnist(is_train=False,
                  flatten=True)

x, y = x.to(device), y.to(device)

test(model, x[:20], y[:20], to_be_shown=True)


# In[ ]:





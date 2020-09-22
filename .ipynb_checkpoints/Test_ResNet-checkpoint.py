#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn


# In[4]:


import sys
import numpy as np
import matplotlib.pyplot as plt

from mnist_classification.data_loader import load_mnist
from mnist_classification.models import resnet_model


# In[6]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# In[7]:


def load(fn, device):
    d = torch.load(fn, map_location=device)
    
    return d['config'], d['model']


# In[8]:


def plot(x, y_hat):
    for i in range(x.size(0)):
        img = (np.array(x[i].detach().cpu(), dtype='float')).reshape(28,28)

        plt.imshow(img, cmap='gray')
        plt.show()
        print("Predict:", float(torch.argmax(y_hat[i], dim=-1)))


# In[9]:


from train import get_model

train_config, state_dict = load(model_fn, device)

model = get_model(train_config).to(device)
model.load_state_dict(state_dict)

print(model)


# In[ ]:





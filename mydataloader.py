import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def data_loader(train_data, test_data = None , test_size= None, batch_size = 32):
    train_loader =  DataLoader(train_data, batch_size = batch_size , shuffle = True)
    
    if(test_data == None and test_size== None):
        
        data_len = len(train_data)
        indices = list(range(data_len))
        #np.random.shuffle(indices)
        split1 = int(np.floor(0.2 * data_len))
        valid_idx , test_idx = indices[:split1], indices[split1:]
        test_sampler = SubsetRandomSampler(valid_idx)
        test_loader = DataLoader(train_data, batch_size= batch_size, sampler=test_sampler)
        dataloaders = {'train':train_loader,'test':test_loader}
        
        ###################
        kfold = []

        for i in range(5):
            data = train_loader
            data_len = len(data)
            indices = list(range(data_len))
            split2 = int(np.floor(0.25 * data_len))
            #train1_idx, valid_idx, train2_idx = indices[:split2 * i], indices[split2 * i : split2 * (i+1)], indices[split2 * (i+1):]
            valid_idx= indices[split2 * i : split2 * (i+1)]
            valid_sampler = SubsetRandomSampler(valid_idx)
            valid_loader = DataLoader(data, batch_size=batch_size, sampler = valid_sampler)
            train_loader = data
            dataloaders = {"train":train_loader, "val":valid_loader, "test":test_loader}
            kfold.append(dataloaders)
            
        return kfold
    
    if(test_data == None and test_size!=None):
        data_len = len(train_data)
        indices = list(range(data_len))
        np.random.shuffle(indices)
        split1 = int(np.floor(test_size* data_len))
        valid_idx , test_idx = indices[:split1], indices[split1:]
        valid_sampler = SubsetRandomSampler(valid_idx)
        valid_loader = DataLoader(train_data, batch_size= batch_size, sampler=valid_sampler)
        dataloaders = {'train':train_loader,'test':valid_loader}         
            
        return dataloaders
    
    if(test_data != None and test_size!=None):
        data_len = len(test_data)
        indices = list(range(data_len))
        np.random.shuffle(indices)
        split1 = int(np.floor(test_size* data_len))
        valid_idx , test_idx = indices[:split1], indices[split1:]
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        valid_loader = DataLoader(test_data, batch_size= batch_size, sampler=valid_sampler)
        test_loader = DataLoader(test_data, batch_size= batch_size, sampler=test_sampler)
        dataloaders = {'train':train_loader,'val':valid_loader,'test':test_loader}
        
        return dataloaders
    
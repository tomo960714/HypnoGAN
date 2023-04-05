# -*- coding: UTF-8 -*-
import numpy as np
import torch

class TimeGAN_Dataset(torch.utils.data.dataset):
    """A time series dataset for TimeGAN.
    Args:
        data(numpy.ndarray): the padded dataset to be fitted. Has to transform to ndarray from DataFrame during initializ
        time(numpy.ndarray): the length of each data
    Parameters:
        - x (torch.FloatTensor): the real value features of the data
        - t (torch.LongTensor): the temporal feature of the data
    """
    def __init__(self,args,data):
        #sanity check data and time
        value = data['Sleeping stage'].values
        time = data['time'].values
        if len(value) != len(time):
            raise ValueError( f"len(value) `{len(value)}` != len(time) {len(time)}")
        if isinstance(time,type(None)):
            time = [len(x) for x in data]
        self.X = torch.FloatTensor(value)
        self.T = torch.LongTensor(time)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx],self.T[idx]
    
    def collate_fn(self, batch):
        """Minibatch sampling
        """
        # Pad sequences to max length
        X_mb = [X for X in batch[0]]
        
        # The actual length of each data
        T_mb = [T for T in batch[1]]
        
        return X_mb, T_mb
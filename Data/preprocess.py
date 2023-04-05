""" 
Preprocess methods used on the datasets

Path: preprocess.py

(0) 

"""

# Local packages:
from typing import List,Tuple
import os
import warnings
warnings.filterwarnings("ignore")

# 3rd party packages:
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


# personal packages:
#from utils import *
from Data import data_loading as ld

def preprocess_data(args):
    """
    padding_value: int = -1.0,
    data_limit: int = None"""
    # Load and preprocess data
    # 
    # 1. Load data from files (csv,mat,xml)
    # 2. Preprocess data:
    # 2.1. Remove outliers
    # 2.2. Extract sequence length and time
    # 2.3. Resample data
    # 2.4. Normalize data
    # 2.5. Padding 
    #  
    # Args:
    #     data_limit (int): The number of lines to load from the data file
    #     padding_value (int): The value used for padding
    #     
    # 
    # Returns:
    #     prep_data (pandas.DataFrame): The processed data

    #######################################
    # 1. Load data from files (csv,mat,xml)
    #######################################

    loaded_data = ld.load_data(data_limit=args.data_limit)
    """
    loaded data =       time_data   , data  , length
    (pandas.DataFrame), (np.array)  ,(list) ,(int)
    
    ()
    """
    #######################################
    # 2. Preprocess data:
    #######################################
    # 2.1. Remove outliers
    #######################################
    """
    Remove row's with unacceptable sleep stages values
    """
    sleep_stages = np.array([1,2,3,4,5])
    loaded_data[loaded_data['data'].apply(lambda x: all(elem in sleep_stages for elem in x))]

    #######################################
    # 2.2. Extract sequence length and time
    #######################################
    """
    Extract sequence length of all lines and time of each line
    """
    loaded_data['length'] = loaded_data['data'].apply(lambda x: len(x))

    
    #######################################
    # 2.4. Normalize data
    #######################################
    """
    Normalize data to [0,1] using MinMaxScaler algorithm
    """

    

    if args.norm_enable == True:
        loaded_data['data']=MinMaxNormalizer(loaded_data['data'])
    

    #######################################
    # 2.5. Padding
    #######################################
    """
    Padding data to given length
    """
   
    # Question: Current padding value is 0, is it ok? Do we need it or just resample?
    data_info = {
        'length' : len(loaded_data),
        'max_length' : max(loaded_data['length']),
        'paddding_value' : args.padding_value,

    }
    
    loaded_data['data'] = loaded_data['data'].apply(lambda x: np.transpose(x))
    prep_data = pd.DataFrame(columns=['time','data'])
    for i in tqdm(range(data_info.length)):
        #create empty array with padding value
        tmp_array = np.empty([data_info.max_length,1])
        tmp_array.fill(args.padding_value)
        #fill array with data
        tmp_array[:loaded_data['data'][i].shape[0],:loaded_data['data'][i].shape[1]] = loaded_data['data'][i]
        #append to prep_data
        prep_data.append(tmp_array)

    return prep_data.to_numpy(),loaded_data['time_data'].to_numpy(),data_info
    

    
    
    


def MinMaxNormalizer(data,min_value=1,max_value=5):
    numerator = data-min_value
    denominator = max_value-min_value
    norm_data = numerator/denominator
    return norm_data
    
    




       









    




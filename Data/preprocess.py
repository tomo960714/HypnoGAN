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


# personal packages:
from utils import *
from Data import data_loading as ld

def preprocess_data():
    # Load and preprocess data
    # 
    # 1. Load data from files (csv,mat,xml)
    # 2. Preprocess data:
    # 2.1. Remove outliers
    # 2.2. Extract sequence length and time
    # 2.3. Resample data
    # 2.4. Impute missing data
    # 2.5. Normalize data
    # 2.6. Sort dataset
    #  
    # Args:
    #     
    # 
    # Returns:
    #     data (pandas.DataFrame): The preproc

    #######################################
    # 1. Load data from files (csv,mat,xml)
    #######################################

    test_limit = 10
    loaded_data = ld.load_data(data_limit=test_limit)

    #######################################
    # 2. Preprocess data:
    #######################################
    # 2.1. Remove outliers
    #######################################

    




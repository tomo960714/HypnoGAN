""" 
Preprocess methods used on the datasets

Path: preprocess.py

(0) MinMaxScaler: Min Max normalizer

"""

#Necessary Packages:
import numpy as np

def MinMaxScaler(data):
    """ 
    Min Max normalizer

    Args:
        data (pandas.DataFrame): The data to be normalized.

    Returns:
        data_normalized (pandas.DataFrame): The normalized data.
    """
    #Initialize the output
    
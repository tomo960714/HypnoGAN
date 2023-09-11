import numpy as np
import ast
import pandas as pd

##################
#Hypnogram manipulating functions
#################

def deconstruct_array(array):
    """Dismantle ID x length x 1 array to ID x length x 6 based on x1 values."""    
    new_array = np.zeros((array.shape[0],array.shape[1],6))
    i_max = array.shape[0]-1
    j_max = array.shape[1]-1
    for i in range(i_max):
        for j in range(j_max):
            new_array[i,j,int(array[i,j,0])] = 1
    return new_array

def reconstruct_array(array):
    """Remake ID x length x 6 array to ID x length x 1 based on x1 values."""
  
    new_array = np.zeros((array.shape[0],array.shape[1],1))
    i_max = array.shape[0]-1
    j_max = array.shape[1]-1
    for i in range(i_max):
        for j in range(j_max):
            new_array[i,j,0] = np.argmax(array[i,j,:])
    return new_array

def flip_hypnogram_y(hypnogram):
    """
    Function to flip hypnograms.
    Args:
        Inputs:
            -hypnogram: a batch or a single hypnogram as numpy array
        Outputs:
            -hypnogram_reordered: flipped hypnogram (1->5, 2->4,4->2,5->1)
    """
    mapping = {1:5,2:4,4:2,5:1}
    hypnogram_reordered = np.where(np.isin(hypnogram, list(mapping.keys())), np.vectorize(mapping.get)(hypnogram), hypnogram)
    return hypnogram_reordered

def flip_hypnogram_x_row(hypnogram,time,idx=None):
    
    if hypnogram.shape[-1] == 6:
        hypnogram = reconstruct_array(hypnogram)
    if idx is not None:
        hypnogram=np.squeeze(hypnogram[idx][0])
    """flipped_hypnogram = []
    
    for i in range(len(hypnogram)):
        flipped_hypnogram.appened(hypnogram[i])
    """
    #for array if time is the same length for all, can't find a solution for different lengths
    #a_ = np.flip(a[:,:time], axis=1)
    #flipping x axis for a single row
    flipped_hypnogram = np.flip(hypnogram[0:time])
    return flipped_hypnogram

def flip_hypnogram_y_row(hypnogram):
    """Function to flip hypnograms.
    Args:
        Inputs:
            -hypnogram: a batch or a single hypnogram as numpy array
        Outputs:
            -hypnogram_reordered: flipped hypnogram (1->5, 2->4,4->2,5->1)
    """
    if hypnogram.shape[-1] == 6:
        hypnogram=reconstruct_array(hypnogram)
    
    mapping = {0:5,1:4,2:3,3:2,4:1,5:0}
    flipped_hypnogram = np.array([mapping[i] for i in hypnogram])
    return flipped_hypnogram

def flip_hypnogram_y_batch(hypnogram):
    """Function to flip a batch of hypnograms.
    Args:
        Inputs:
            -hypnogram: a batch or a single hypnogram as numpy array
        Outputs:
            -hypnogram_reordered: flipped hypnogram (1->5, 2->4,4->2,5->1)
    """
    if hypnogram.shape[-1] == 6:
        hypnogram = reconstruct_array(hypnogram)
    
    mapping = {0:5,1:4,2:3,3:2,4:1,5:0}
    flipped_hypnogram = np.vectorize(mapping.get)(hypnogram)
    return flipped_hypnogram

def remove_padding(hypnogram):
    # Remove padding (zeros) from each array and convert to a list
    trimmed_list = [np.trim_zeros(arr, 'b') for arr in hypnogram]
    return trimmed_list

def string_to_float_list(s):
    return [float(x) for x in s.strip('[]').split(',')]
def string_to_int_list(s):
    return [int(x) for x in s.strip('[]').split(',')]
def string_to_float_array(s):
    """Convert array-as-string to float array"""
    return [float(x) for x in ast.literal_eval(s)]
def string_to_int_array(s):
    """Convert array-as-string to int array"""
    return [int(x) for x in ast.literal_eval(s)]

def load_from_csv(filename):
    main_dataset = pd.DataFrame()
    main_dataset = pd.read_csv(filename,sep=';',index_col=0)
    main_dataset['Sleeping_stage'] = main_dataset['Sleeping_stage'].apply(string_to_int_array)
    sleeping_stage_array = main_dataset['Sleeping_stage'].to_numpy()
    numpy_array = np.array([np.array(lst) for lst in sleeping_stage_array])
    fake_data = numpy_array.reshape((numpy_array.shape[0], numpy_array.shape[-1], 1))
    return fake_data
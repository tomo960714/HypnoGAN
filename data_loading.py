
#imports
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import pandas as pd

#local imports
from utils import load_mat_as_df

def DataLoader(data_limit=None):
    """
    Load and preprocess real life datasets.
    
    Args:


    Returns:
        dataset (pandas.DataFrame): The dataset.
    
    """

    dataset = pd.DataFrame()

    
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filenames = askopenfilenames() # show an "Open" dialog box and return the path to the selected file
    print(filenames)
    for filename in filenames:
        
        if filename.endswith('.mat'):
            #output df format: [id,value_array]
            df = load_mat_as_df(filename)
            print(df)
        
        elif filename.endswith('.csv'):
            ## TODO: add csv support
            #df = 
            pass

        elif filename.endswith('.xml'):
            ## TODO: add xml support
            #df =
            pass
        else:
            pass
    
        dataset.append(df)

    #Cut df to data_limit size for testing purposes
    if data_limit is not None:
        dataset = dataset[:data_limit]
    
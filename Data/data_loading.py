
#imports
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import pandas as pd
import pandas as pd
import scipy.io as sio
import tkinter as tk
import numpy as np

#local imports


def load_data(data_limit=None):
    """
    Load and preprocess real life datasets.
    
    Args:
        data_limit (int): The number of data points to load. If None, all data points are loaded. Default: None. Used for testing.


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

            """ CSV format:
            ID|refresh_rate|value_array|length|additional_info

            """
            pass

        elif filename.endswith('.xml'):
            ## TODO: add xml support
            #df =
            pass
        else:
            print("Unsupported file format, skipping file:",filename,".")
            pass
    
        dataset.append(df)

    #Cut df to data_limit size for testing purposes
    if data_limit is not None:
        dataset = dataset[:data_limit]
    
    return dataset #dataset as df

def load_mat_as_df(mat_file_path, var_name):
    mat = sio.loadmat(mat_file_path,simplify_cells=True)

    if var_name not in list(mat.keys()):
        var_name = get_variable_name(mat)   
        

    return pd.DataFrame(mat[var_name])

def get_variable_name(loaded_mat):

    
    root = tk.Tk()
    root.title('.mat variable selector')
    tk.Label(root, text="Choose a variable:").pack()
    choices = list(loaded_mat.keys())

    variable = tk.StringVar(root)
    variable.set(choices[0]) # default value
    w = tk.Combobox(root, textvariable=variable,values=choices)

    w.pack()
    def ok():
        print ("value is:" + variable.get())
        root.destroy()
    def cancel():
        root.destroy()
        raise InterruptedError('User cancelled, invalid variable name')

    button1 = tk.Button(root, text="OK", command=ok)
    button2 = tk.Button(root, text="Cancel", command=cancel)
    button1.pack()
    button2.pack()
    root.mainloop()
    
    return variable.get()
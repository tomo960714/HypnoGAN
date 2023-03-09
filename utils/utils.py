import pandas as pd
import scipy.io as sio
from tkinter import *
import numpy as np

def load_mat_as_df(mat_file_path, var_name):
    mat = sio.loadmat(mat_file_path,simplify_cells=True)

    if var_name not in list(mat.keys()):
        var_name = get_variable_name(mat)   
        

    return pd.DataFrame(mat[var_name])

def get_variable_name(loaded_mat):

    
    root = Tk()
    root.title('.mat variable selector')
    Label(root, text="Choose a variable:").pack()
    choices = list(mat.keys())

    variable = StringVar(root)
    variable.set(choices[0]) # default value
    w = Combobox(root, textvariable=variable,values=choices)

    w.pack()
    def ok():
        print ("value is:" + variable.get())
        root.destroy()
    def cancel():
        root.destroy()
        raise InterruptedError('User cancelled, invalid variable name')

    button1 = Button(root, text="OK", command=ok)
    button2 = Button(root, text="Cancel", command=cancel)
    button1.pack()
    button2.pack()
    root.mainloop()
    
    return variable.get()

def sin_noise(ns,seq_len,dim,offset = 0.0,amp = 1.0):
    """
    Sine wave generation.
    
    Args:
        ns (int): Number of sequences.
        seq_len (int): Length of each sequence.
        dim (int): Feature dimensions.
    
    Returns:
        data (list): The generated data.
    """
    
    # Initialize the output
    data = list()

    #Generate sin data
    for i in range(ns):
        #initialize time series:
        ts = list()
        #each feature
        for j in range(dim):
            #generate random frequency
            freq = np.random.uniform(0,0.1)
            #generate random phase
            phase = np.random.uniform(0,0.1)

            #adding random amplitude option
            if amp == 'random':
                #generate random amplitude
                amp = np.random.uniform(0.1,5)
            else:
                amp = float(amp)

            #adding random offset option
            if offset == 'random':        
                #generate random offset
                offset = np.random.uniform(-5,5)
            else:
                offset = float(offset)
            #generate time series
            ts_data= [offset + amp*np.sin(j*freq+phase) for j in range(seq_len)]
            ts.append(ts_data)

        #align row/column
        ts = np.transpose(np.asarray(ts))
        #normalize to [0,1]
        ts = (ts + 1) * 0.5
        #append to data
        data.append(ts)
    
    return data





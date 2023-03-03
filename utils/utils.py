import pandas as pd
import scipy.io as sio
from tkinter import *

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
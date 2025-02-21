
#imports
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import pandas as pd
import pandas as pd
import scipy.io as sio
import tkinter as tk
import numpy as np

#local imports


def load_data(args):
    """
    data_limit=None,save_dataset=None
    Load and preprocess real life datasets.
    
    Args:
        data_limit (int): The number of data points to load. If None, all data points are loaded. Default: None. Used for testing.
        save_dataset (bool): If 'Full', the dataset is saved to a csv file. If it's 'limited', than save the limited dataset if data_limit is not None. Default: None.


    Returns:
        dataset (pandas.DataFrame): The dataset.
    
    """

    main_dataset = pd.DataFrame()

    
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filenames = askopenfilenames() # show an "Open" dialog box and return the path to the selected file
    print(filenames)
    for filename in filenames:
        
        if filename.endswith('.mat'):
            #output df format: [id,value_array]
            df = load_mat_as_df(filename)
            print(df)
        
        elif filename.endswith('.csv'):

            #use create_dataset_csv.py to create a csv file
            if filename.find('dataset') != -1:
                df = pd.read_csv(filename,sep=';',index_col=0)

            """ CSV format:
            ID|time|Sleeping stage|length|additional_info

            """
            pass

        elif filename.endswith('.xml'):
            ## TODO: add xml support
            #df =
            pass
        else:
            print("Unsupported file format, skipping file:",filename,".")
            pass
    
        main_dataset.append(df)

    #Cut df to data_limit size for testing purposes
    if args.data_limit is not None:
        if args.save_dataset == 'Full':
            #save dataset to a csv file
            main_dataset.to_csv('Full_dataset.csv',sep=';')
        
        elif args.save_dataset == 'Limited':
            main_dataset = main_dataset[:args.data_limit]
            #save dataset to a csv file
            main_dataset.to_csv('limited_dataset.csv',sep=';')
        elif args.save_dataset == 'None':
            main_dataset = main_dataset[:args.data_limit]
            pass
        else:
            raise ValueError("Invalid save_dataset value, valid values are 'Full','Limited','None'.")

    elif args.data_limit is None:
        if args.save_dataset == 'Full':
            #save dataset to a csv file
            main_dataset.to_csv('full_dataset.csv',sep=';')
        elif args.save_dataset == 'Limited':
            print("Warning: data_limit is None, dataset is not limited, saving full dataset.")
            main_dataset.to_csv('full_dataset.csv',sep=';')
        elif args.save_dataset == 'None':
            pass
        else:
            raise ValueError("Invalid save_dataset value, valid values are 'Full','Limited','None'.")

    
    return main_dataset #dataset as df

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
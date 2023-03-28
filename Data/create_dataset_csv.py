# %%

# 3rd party packages:
import pandas as pd
import numpy as np
from tkinter import Tk
from tkinter import filedialog
# local packages:
import os

# personal packages:


# %%
# get folder path
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
folder_path = filedialog.askdirectory()

if "o1" not in folder_path:
    if os.path.exists(folder_path + "/o1"):
        folder_path = folder_path + "/o1"
        print(folder_path)
    else:
        print("No o1 folder found")
        raise FileNotFoundError
else:
    print(folder_path)

# %%
# get all files in folder
files = os.listdir(folder_path)
print(files)
used_cols = [3,5]

#new_dataset = pd.DataFrame(columns=['name','time','value'])
#temp_df = pd.DataFrame(columns=['time','value'])

new_names_df = pd.DataFrame(files, columns=['Name'])
new_names_df['time'] = pd.Series(dtype='float64')
new_names_df['value'] = pd.Series(dtype='int64')
#result = pd.concat([new_dataset, new_names_df], ignore_index=True)

# %%
for index,file in enumerate(files):
    temp_df = pd.read_csv(os.path.join(folder_path,file,'STAGE_E.txt'),skiprows=1 ,usecols=used_cols,delimiter='\t',names=['time','value'])
    data = np.loadtxt(os.path.join(folder_path,file,'STAGE_E.txt'), delimiter='\t', skiprows=1, usecols=used_cols)
    new_names_df['time'].loc[index] = data[:,0]
# %%

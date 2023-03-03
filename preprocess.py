#imports
from tkinter import Tk
from tkinter.filedialog import askopenfilenames

#local imports
from utils import load_mat_as_df

def preprocess():

    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilenames() # show an "Open" dialog box and return the path to the selected file
    print(filename)

    

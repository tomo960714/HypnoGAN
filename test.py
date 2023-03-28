import tkinter as tk
from tkinter import filedialog

# Create a Tkinter root window
root = tk.Tk()

# Hide the root window
root.withdraw()

# Define a function to handle the button click event
def select_folders():
    # Open a directory dialog box and retrieve the selected directory paths
    dir_paths = filedialog.askdirectory()
    # Show the selected directory paths
    print(dir_paths)

# Create a button to select folders
button = tk.Button(root, text="Select Folders", command=select_folders)
button.pack()

# Run the main loop of the Tkinter window
root.mainloop()

#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from manipulate import reconstruct_array, load_from_csv
import pandas as pd
#%%
exp = "histogram"
filename = "fake_data.csv"
fake_data = load_from_csv(filename)


# %%
#sum by column
#print(fake_data)
#print(sum_by_col)
#fake_data = fake_data.reshape((fake_data.shape[0], fake_data.shape[1]))
fake_data = np.squeeze(fake_data)
print(fake_data.shape)
#%% 
saved_fake_data = fake_data

#%%
fake_data=saved_fake_data


#%%
print(fake_data.shape)
#cut the first 5 rows, and the last 5 rows into a new array
new_data = np.concatenate((fake_data[:, :5],fake_data[:,759:764],fake_data[:,-6:-1]),axis=1)
#new_data = np.concatenate((new_data,))

print(f'new data: {new_data.shape}')

new_data_save=new_data
# %%
new_data=new_data_save


#%%
num_rows, num_columns = new_data.shape


#%%
fig, axs = plt.subplots(1, num_columns, figsize=(6 * num_columns,8 ))
print(num_columns)
# Iterate over data points
x = np.arange(6)
print(x)
for i in range(num_columns):
    # Create histogram for the current data point
    #axs[i].hist(new_data[:,i], bins=6, edgecolor='black', density=True)
    #axs[i].tick_params(axis='y', labelsize=20)  # Set the font size of y-axis labels
    #axs[i].set_xlim(0, 6)
    value_counts = np.bincount(new_data[:, i])
    #reshape the array to 7 rows
    #value_counts = value_counts.reshape((6))
    print(value_counts)
    axs[i].bar(x, value_counts)
    
# put subplots close to each other
plt.tight_layout()


# Display the plot
plt.show()

# %%

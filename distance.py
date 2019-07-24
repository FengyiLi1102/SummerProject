# import libraries for analysis
import pandas as pd
import math

# read data from local address
train_df_full = pd.read_csv(r'\\icnas4.cc.ic.ac.uk\fl4718\Desktop\Machine learning\Data\train.csv', index_col=0)
structures_df_full = pd.read_csv(r'\\icnas4.cc.ic.ac.uk\fl4718\Desktop\Machine learning\Data\structures.csv')

# Make a copy of  the data for avoiding changing the original data
train_df_copy = train_df_full.copy()

# Initial the molecule name for check in the loop. Randomly chooce a value.
molecule_name = '0'

# loop for finding the distance
for index in train_df_copy.index:

    # Avoid for repeating making sub dataframe for the same molecule
    current_molecule_name = train_df_full.loc[index, 'molecule_name']
    if current_molecule_name == molecule_name:
        current_sub_df = sub_df
    else:
        current_sub_df = structures_df_full[structures_df_full.molecule_name == current_molecule_name].reset_index()
    
    # Get indexes for two spins
    index_0 = train_df_full.loc[index, 'atom_index_0']
    index_1 = train_df_full.loc[index, 'atom_index_1']

    # Get x, y, z components for further calculation
    x_component = current_sub_df.loc[index_0, 'x'] - current_sub_df.loc[index_1, 'x']
    y_component = current_sub_df.loc[index_0, 'y'] - current_sub_df.loc[index_1, 'y']
    z_component = current_sub_df.loc[index_0, 'z'] - current_sub_df.loc[index_1, 'z']

    # Calculate the distance
    distance = math.sqrt(x_component**2 + y_component**2 + z_component**2)

    # Add one more column to the train data for showing the distance
    train_df_copy.loc[index, 'distance'] = distance

    # Assign two values for the check at the beginning of the loop
    molecule_name = current_molecule_name
    sub_df = current_sub_df

# Write to a new csv with name 'train_distance.csv'
pd.to_csv('train_distance.csv')
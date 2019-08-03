#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# libraries required
import pandas as pd
import numpy as np
from tqdm import tqdm


def reduce_memory(df, verbose=True):
    """
    :param: df - dataframe required to decrease the memory usage
    :param: verbose - show logging output if 'Ture'

    Goal: Reduce the memory usage by decreasing the type of the value if applicable

    Return: original dataframe with lower memory usage
    """

    numerics = ['int64', 'int16', 'int32', 'float64', 'float32', 'float16']
    start_memory = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_memory = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_memory, 100 * (start_memory - end_memory) / start_memory))

    return df


def struc1_merge(df1, df2, index):
    """
    :param: df1 - training data
    :param: df2 - structure data after being added electronegativity, radius, bond_lengths, hybridization, surrounding atoms (bonds),
            position info. (x, y, z)
    :param: index - atom_index in the coupling

    Goal: Merge original training dataframe with processed structure data to form a new dataframe for further training process

    Return: Merged dataframe
    """

    struc1_train_merge = pd.merge(df1, df2, how='left',
                                  left_on=['molecule_name', f'atom_index_{index}', f'atom_{index}', f'x_{index}', f'y_{index}', f'z_{index}'],
                                  right_on=['molecule_name', 'atom_index', 'atom', 'x', 'y', 'z'])
    
    struc1_train_merge = struc1_train_merge.drop(['n_bonds'], axis=1)
    
    struc1_train_merge = struc1_train_merge.rename(columns={'EN': f'EN_{index}',
                                                            'RD': f'RD_{index}',
                                                            'bond_lengths': f'bond_lengths_{index}',
                                                            'hybri': f'hybri_{index}',
                                                            'bonds': f'bonds_{index}',
                                                            'pi_bonds': f'pi_bonds_{index}'})
    
    return struc1_train_merge


def n_bonds(structures):
    """
    :param: structures - structure.csv from local data
    
    Goal: Calculate the number of bonds for each molecule.

    Return: Structure dataframe with number of bonds (n_bonds) and lists consisting of indexes of connecting atoms (bonds)
    """

    i_atom = structures['atom_index'].values
    p = structures[['x', 'y', 'z']].values
    p_compare = p
    m = structures['molecule_name'].values
    m_compare = m
    r = structures['RD'].values
    r_compare = r

    source_row = np.arange(len(structures))
    max_atoms = 28

    bonds = np.zeros((len(structures)+1, max_atoms+1), dtype=np.int8)
    bond_dists = np.zeros((len(structures)+1, max_atoms+1), dtype=np.float32)

    print('Calculating bonds')

    for i in tqdm(range(max_atoms-1)):
        p_compare = np.roll(p_compare, -1, axis=0)
        m_compare = np.roll(m_compare, -1, axis=0)
        r_compare = np.roll(r_compare, -1, axis=0)

        mask = np.where(m == m_compare, 1, 0) #Are we still comparing atoms in the same molecule?
        dists = np.linalg.norm(p - p_compare, axis=1) * mask
        r_bond = r + r_compare

        bond = np.where(np.logical_and(dists > 0.0001, dists < r_bond), 1, 0)

        source_row = source_row
        target_row = source_row + i + 1 #Note: Will be out of bounds of bonds array for some values of i
        target_row = np.where(np.logical_or(target_row > len(structures), mask==0), len(structures), target_row) #If invalid target, write to dummy row

        source_atom = i_atom
        target_atom = i_atom + i + 1 #Note: Will be out of bounds of bonds array for some values of i
        target_atom = np.where(np.logical_or(target_atom > max_atoms, mask==0), max_atoms, target_atom) #If invalid target, write to dummy col

        bonds[(source_row, target_atom)] = bond
        bonds[(target_row, source_atom)] = bond
        bond_dists[(source_row, target_atom)] = dists
        bond_dists[(target_row, source_atom)] = dists

    bonds = np.delete(bonds, axis=0, obj=-1) #Delete dummy row
    bonds = np.delete(bonds, axis=1, obj=-1) #Delete dummy col
    bond_dists = np.delete(bond_dists, axis=0, obj=-1) #Delete dummy row
    bond_dists = np.delete(bond_dists, axis=1, obj=-1) #Delete dummy col

    print('Counting and condensing bonds')

    bonds_numeric = [[i for i,x in enumerate(row) if x] for row in tqdm(bonds)]
    bond_lengths = [[dist for i,dist in enumerate(row) if i in bonds_numeric[j]] for j,row in enumerate(tqdm(bond_dists))]
    n_bonds = [len(x) for x in bonds_numeric]

    #bond_data = {'bond_' + str(i):col for i, col in enumerate(np.transpose(bonds))}
    #bond_data.update({'bonds_numeric':bonds_numeric, 'n_bonds':n_bonds})

    bond_data = {'bonds':bonds_numeric, 'n_bonds':n_bonds, 'bond_lengths':bond_lengths}
    bond_df = pd.DataFrame(bond_data)
    structures = structures.join(bond_df)
    
    return structures


def struc_merge(df, struc, index):
    """
    :param: df - The dataframe to be merged with structure data
    :param: struc - structure data
    :param: index - index of atom in the coupling

    Goal: Merger two dataframe.

    Return: a new dataframe after merged
    """

    # Merge train and structures data based on the atom index
    df_struc = pd.merge(df, struc, how='left', 
                        left_on=['molecule_name', f'atom_index_{index}'], 
                        right_on=['molecule_name', 'atom_index'])

    # Drop the atom index column
    df_struc = df_struc.drop('atom_index', axis=1)

    # Rename the columns
    df_struc = df_struc.rename(columns={'atom': f'atom_{index}',
                                        'x': f'x_{index}',
                                        'y': f'y_{index}',
                                        'z': f'z_{index}'})

    return df_struc


def distance(df, structures):
    """
    :param: df - Data that need to calculate distance

    Goal: Calculate the distance between two spins

    Return: DataFrame with distance added
    """

    # Make a copy of  the data for avoiding changing the original data
    df_copy = df.copy()

    # Merge data
    df_copy = struc_merge(df_copy, structures, 0)
    df_copy = struc_merge(df_copy, structures, 1)

    get_ipython().run_line_magic('time', '')
    # This block for calculating the distance between two spins
    df_p_0 = df_copy[['x_0', 'y_0', 'z_0']].values
    df_p_1 = df_copy[['x_1', 'y_1', 'z_1']].values

    df_copy['distance'] = np.linalg.norm(df_p_0 - df_p_1, axis=1)

    return df_copy


def hybridization(structures):
    """
    :param: structures - structures data

    Goal: Calculate each hybridization in the structures data

    Return: structure data with hybridization column added
    """
    
    # 'C' has different types of hybridizations with different number of bonds.
    # '4' for four bonds
    hybri_dict = {'C': {'4': 3, '3': 2, '2': 2, '1': 0},
                  'N': {'4': 0, '3': 3, '2': 2, '1': 1},
                  'O': {'2': 2, '1': 1},
                  'H': {'1': 0},
                  'F': {'1': 0}}
                # 3 bonds- sp3, 2 - sp2, 1 - sp
    
    hybri = []

    for i in tqdm(range(len(structures))):
        hybri.append(hybri_dict[structures.loc[i, 'atom']][str(structures.loc[i, 'n_bonds'])])
    
    structures['hybri'] = hybri

    return structures


def pi_bonds(structures):
    """
    :param: structures - structures data

    Goal: Calculate the number of pi_bonds for each atom

    Return: structures with pi_bonds column added
    """

    # The number of atoms connecting to an atom is related with the number of pi bonds.
    # Eg: In 'C', if there are 4 bonds around, then the number of pi bonds is 0.
    pi_bond = {'C': {'4': 0, '2': 2, '3': 1},
               'N': {'4': 0, '3': 0, '2': 1, '1': 2},
               'O': {'1': 1, '2': 0},
               'H': {'1': 0},
               'F': {'1': 0}}

    pi_bond_ = []

    for i in tqdm(range(len(structures))):
        pi_bond_.append(pi_bond[structures.loc[i, 'atom']][str(structures.loc[i, 'n_bonds'])])

    structures['pi_bonds'] = pi_bond_

    return structures


def electronegativity(atom_name, structures):
    """
    :param: atom_name - list or np.ndarray consisting of name of atoms
    :param: structures - structures data

    Goal: Assign an electrinegativity for each atom

    Return: structures with electrineativity column added
    """

    electronegativity = {'H':2.2, 'C':2.55, 'N':3.04, 'O':3.44, 'F':3.98}
    en_ = [electronegativity[x] for x in tqdm(atom_name)]

    structures['EN'] = en_

    return structures


def radius(atom_name, structures):
    """
    :param: atom_name - list or np.ndarray consisting of name of atoms
    :param: structures - structures data

    Goal: Assign an radius for each atom

    Return: structures with radius column added
    """

    atomic_radius = {'H':0.38, 'C':0.77, 'N':0.75, 'O':0.73, 'F':0.71} # Without fudge factor

    fudge_factor = 0.05
    atomic_radius = {k:v + fudge_factor for k,v in atomic_radius.items()}
    rd_ = [atomic_radius[x] for x in atom_name]

    structures['RD'] = rd_

    return structures


def map_atom_info(df_1,df_2, atom_idx):
    """
    :param: df_1 - train data
    :param: df_2 - structure data
    :param: atom_ind - atom index in coupling

    Goal: Merge two dataframe for further using

    Return: A new dataframe after merged
    """

    df = pd.merge(df_1, df_2, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    df = df.drop('atom_index', axis=1)

    return df


def create_closest(df_train):
    df_temp=df_train.loc[:,["molecule_name","atom_index_0","atom_index_1","distance","x_0","y_0","z_0","x_1","y_1","z_1"]].copy()
    df_temp_=df_temp.copy()
    df_temp_= df_temp_.rename(columns={'atom_index_0': 'atom_index_1',
                                       'atom_index_1': 'atom_index_0',
                                       'x_0': 'x_1',
                                       'y_0': 'y_1',
                                       'z_0': 'z_1',
                                       'x_1': 'x_0',
                                       'y_1': 'y_0',
                                       'z_1': 'z_0'})

    df_temp=pd.concat(objs=[df_temp,df_temp_],axis=0)

    df_temp["min_distance"]=df_temp.groupby(['molecule_name', 'atom_index_0'])['distance'].transform('min')
    df_temp= df_temp[df_temp["min_distance"]==df_temp["distance"]]

    df_temp=df_temp.drop(['x_0','y_0','z_0','min_distance'], axis=1)
    df_temp= df_temp.rename(columns={'atom_index_0': 'atom_index',
                                     'atom_index_1': 'atom_index_closest',
                                     'distance': 'distance_closest',
                                     'x_1': 'x_closest',
                                     'y_1': 'y_closest',
                                     'z_1': 'z_closest'})

    for atom_idx in [0,1]:
        df_train = map_atom_info(df_train,df_temp, atom_idx)
        df_train = df_train.rename(columns={'atom_index_closest': f'atom_index_closest_{atom_idx}',
                                            'distance_closest': f'distance_closest_{atom_idx}',
                                            'x_closest': f'x_closest_{atom_idx}',
                                            'y_closest': f'y_closest_{atom_idx}',
                                            'z_closest': f'z_closest_{atom_idx}'})
    return df_train


def add_cos_features(df):
    """
    :param: df - dataframe containing necessary data for calculating the cosine value

    Goal: Calculating cosine value

    Return: dataframe with cosine data added
    """

    # The modulus of the 
    df["distance_0"]=((df['x_0']-df['x_closest_0'])**2+(df['y_0']-df['y_closest_0'])**2+(df['z_0']-df['z_closest_0'])**2)**(1/2)
    df["distance_1"]=((df['x_1']-df['x_closest_1'])**2+(df['y_1']-df['y_closest_1'])**2+(df['z_1']-df['z_closest_1'])**2)**(1/2)
    
    # Unit vector along each direction
    df["vec_0_x"]=(df['x_0']-df['x_closest_0'])/df["distance_0"]
    df["vec_0_y"]=(df['y_0']-df['y_closest_0'])/df["distance_0"]
    df["vec_0_z"]=(df['z_0']-df['z_closest_0'])/df["distance_0"]
    df["vec_1_x"]=(df['x_1']-df['x_closest_1'])/df["distance_1"]
    df["vec_1_y"]=(df['y_1']-df['y_closest_1'])/df["distance_1"]
    df["vec_1_z"]=(df['z_1']-df['z_closest_1'])/df["distance_1"]
    
    # Ratio between the difference along each direction to the distance
    df["vec_x"]=(df['x_1']-df['x_0'])/df["distance"]
    df["vec_y"]=(df['y_1']-df['y_0'])/df["distance"]
    df["vec_z"]=(df['z_1']-df['z_0'])/df["distance"]

    # Cosine of each component
    df["cos_0_1"]=df["vec_0_x"]*df["vec_1_x"]+df["vec_0_y"]*df["vec_1_y"]+df["vec_0_z"]*df["vec_1_z"]
    df["cos_0"]=df["vec_0_x"]*df["vec_x"]+df["vec_0_y"]*df["vec_y"]+df["vec_0_z"]*df["vec_z"]
    df["cos_1"]=df["vec_1_x"]*df["vec_x"]+df["vec_1_y"]*df["vec_y"]+df["vec_1_z"]*df["vec_z"]

    df=df.drop(['vec_0_x','vec_0_y','vec_0_z','vec_1_x','vec_1_y','vec_1_z','vec_x','vec_y','vec_z'], axis=1)

    # Angle for each component
    df["Angle"] = df["cos_0_1"].apply(lambda x: np.arccos(x)) * 180 / np.pi
    df["cos_0"] = df["cos_0"].apply(lambda x: np.arccos(x)) * 180 / np.pi
    df["cos_1"] = df["cos_1"].apply(lambda x: np.arccos(x)) * 180 / np.pi

    return df


# File paths
train_path = r'\\icnas4.cc.ic.ac.uk\fl4718\Desktop\Machine learning\Data\train.csv'
structures_path = r'\\icnas4.cc.ic.ac.uk\fl4718\Desktop\Machine learning\Data\structures.csv'
test_path = r'\\icnas4.cc.ic.ac.uk\fl4718\Desktop\Machine learning\Data\test.csv'

# read data from local address
train_df_full = pd.read_csv(train_path, index_col=0)
structures_df_full = pd.read_csv(structures_path, dtype={'atom_index': np.int8})
test_df_full = pd.read_csv(test_path)

# Add distance feature to the test and trin data
train_df = distance(train_df_full, structures_df_full)
test_df = distance(test_df_full, structures_df_full)

# ndarray with names of each atom in the structures csv
atom = structures_df_full['atom'].values

# Add electronegativity and radius colmun to the structures csv
structures = electronegativity(atom, structures_df_full)
structures = radius(atom, structures)

# Add number of bonds and connecting atoms columns
structures = n_bonds(structures)

# Add hybridization column
structures = hybridization(structures)

# Add pi_bonds column
structures = pi_bonds(structures)

# Merge structures data and train data
struc_train = struc1_merge(train_df, structures, 0)
struc_train = struc1_merge(struc_train, structures, 1)

struc_train = struc_train.drop(['atom_index_x', 'atom_x', 'x_x', 'y_x', 'z_x',
                                'atom_index_y', 'atom_y','x_y', 'y_y', 'z_y'], axis=1)

# Add bond angle column
struc_train = create_closest(struc_train)
struc_train = add_cos_features(struc_train)

# The list of type for further training
type_list = list(struc_train['type'].unique())

# Drop the target column for training
y = struc_train['scalar_coupling_constant']
struc_train = struc_train.drop(['scalar_coupling_constant'], axis=1)

# Select features for training
X = struc_train[['molecule_name',
                           'type',
                           'distance',
                           'EN_0',
                           'RD_0',
                           'hybri_0',
                           'pi_bonds_0',
                           'EN_1',
                           'RD_1',
                           'hybri_1',
                           'pi_bonds_1',
                           'Angle']]


# import libraries for analysis
import pandas as pd
import numpy as np
from tqdm import tqdm


def struc_merge(df, struc, index):
   """
   :param: df - The dataframe to be merged with structure data
   :param: struc - structure data
   :param: index - index of atom in the coupling

   Goal: Merger two dataframe.

   Return: a new dataframe after merged
   """

    # Merge train and structures data based on the atom index
    df_struc = pd.merge(df, struc, how='left', left_on=['molecule_name', f'atom_index_{index}'], right_on=['molecule_name', 'atom_index'])
    
    # Drop the atom index column
    df_struc = df_struc.drop('atom_index', axis=1)

    # Rename the columns
    df_struc = df_struc.rename(columns={'atom': f'atom_{index}',
                                        'x': f'x_{index}',
                                        'y': f'y_{index}',
                                        'z': f'z_{index}'})

    return df_struc


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
    r = structures['rad'].values
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
        
        mask = np.where(m == m_compare, 1, 0) # Check whether we are comparing atoms in the same molecule
        dists = np.linalg.norm(p - p_compare, axis=1) * mask
        r_bond = r + r_compare
        
        bond = np.where(np.logical_and(dists > 0.0001, dists < r_bond), 1, 0)
        
        source_row = source_row
        target_row = source_row + i + 1 # Note: Will be out of bounds of bonds array for some values of i
        target_row = np.where(np.logical_or(target_row > len(structures), mask==0), len(structures), target_row) # If invalid target, write to dummy row
        
        source_atom = i_atom
        target_atom = i_atom + i + 1 # Note: Will be out of bounds of bonds array for some values of i
        target_atom = np.where(np.logical_or(target_atom > max_atoms, mask==0), max_atoms, target_atom) # If invalid target, write to dummy col
        
        bonds[(source_row, target_atom)] = bond
        bonds[(target_row, source_atom)] = bond
        bond_dists[(source_row, target_atom)] = dists
        bond_dists[(target_row, source_atom)] = dists

    bonds = np.delete(bonds, axis=0, obj=-1) # Delete dummy row
    bonds = np.delete(bonds, axis=1, obj=-1) # Delete dummy col
    bond_dists = np.delete(bond_dists, axis=0, obj=-1) # Delete dummy row
    bond_dists = np.delete(bond_dists, axis=1, obj=-1) # Delete dummy col

    print('Counting and condensing bonds')

    bonds_numeric = [[i for i,x in enumerate(row) if x] for row in tqdm(bonds)]
    bond_lengths = [[dist for i,dist in enumerate(row) if i in bonds_numeric[j]] for j,row in enumerate(tqdm(bond_dists))]
    n_bonds = [len(x) for x in bonds_numeric]

    # bond_data = {'bond_' + str(i):col for i, col in enumerate(np.transpose(bonds))}
    # bond_data.update({'bonds_numeric':bonds_numeric, 'n_bonds':n_bonds})

    bond_data = {'bonds':bonds_numeric, 'n_bonds':n_bonds, 'bond_lengths':bond_lengths}
    bond_df = pd.DataFrame(bond_data)
    structures = structures.join(bond_df)
    
    return structures


def distance(df):
    """
    :param: df - Data that need to calculate distance

    Goal: Calculate the distance between two spins

    Return: DataFrame with distance added
    """

    # Make a copy of  the data for avoiding changing the original data
    df_copy = df.copy()

    # Merge data
    df_copy = struc_merge(df_copy, structures_df_full, 0)
    df_copy = struc_merge(df_copy, structures_df_full, 1)

    %%time
    # This block for calculating the distance between two spins
    df_p_0 = df[['x_0', 'y_0', 'z_0']].values
    df_p_1 = df[['x_1', 'y_1', 'z_1']].values

    df['distance'] = np.linalg.norm(df_p_0 - df_p_1, axis=1)

    return df


def hybridization(structures):
    """
    :param: structures - structures data

    Pre-check:
    The situation that two atoms connecting to a C with one triple and one single bond is checked and there are no such situations 
    in the data set, which is more straight for us to count the number of pi bonds for each molecule.This means that the number of 
    pi bonds in each molecule will be directly related with the type of hybridization. For more restrict consideration, here the 
    pre-check is done.

    Goal: Calculate each hybridization in the structures data

    Return: structure data with hybridization column added
    """
    
    # 'C' has different types of hybridizations with different number of bonds.
    # '4' for four bonds
    hybri = {'C': {'4': 3, '3': 2, '2': 1, '1': 0},
              'N': {'4': 0, '3': 3, '2': 2, '1': 1},
              'O': {'2': 2, '1': 1},
              'H': {'1': 0},
              'F': {'1': 0}}
    
    hybri_ = []

    for i in tqdm(range(len(structures))):
            hybri_.append(hybri[structures.loc[i, 'atom']][str(structures.loc[i, 'n_bonds'])])
    
    structures['hybri'] = hybri_

    return structures


def pi_bonds(structures):
    """
    :param: structures - structures data

    Goal: Calculate the number of pi_bonds for each atom

    Return: structures with pi_bonds column added
    """

    # The number of atoms connecting to an atom is related with the number of pi bonds.
    # Eg: In 'C', if there are 4 bonds around, then the number of pi bonds is 0.
    pi_bond = {'C': {'4': 0, '3': 1, '2': 2},
               'N': {'4': 0, '3': 0, '2': 1, '1': 2},
               'O': {'1': 1, '2': 0},
               'H': {'1': 0},
               'F': {'1': 0}}

    pi_bond_ = []

    for i in tqdm(range(len(structures))):
        pi_bond_.append(pi_bond[structures.loc[i, 'atom']][str(len(structures.loc[i, 'bonds']))])

    structures['pi_bonds'] = pi_bond_

    return structures


def electronegativity(atom_name, structures):
    """
    :param: atom_name - list or np.ndarray consisting of name of atoms
    :param: structures - structures data

    Goal: Assign an electrinegativity for each atom

    Return: structures with electrineativity column added
    """

    en = {'H':2.2, 'C':2.55, 'N':3.04, 'O':3.44, 'F':3.98 }
    en_ = [en[x] for x in tqdm(atom_name)]

    structures['EN'] = en_

    return structures


def radius(atom_name, structures):
    """
    :param: atom_name - list or np.ndarray consisting of name of atoms
    :param: structures - structures data

    Goal: Assign an radius for each atom

    Return: structures with radius column added
    """

    rd = {'H':0.38, 'C':0.77, 'N':0.75, 'O':0.73, 'F':0.71}
    rd_ = [rd[x] for x in tqdem(atom_name)]

    structures['RD'] = rd_

    return structures


# File paths
train_path = r'\\icnas4.cc.ic.ac.uk\fl4718\Desktop\Machine learning\Data\train.csv'
structures_path = r'\\icnas4.cc.ic.ac.uk\fl4718\Desktop\Machine learning\Data\structures.csv'
test_path = r'\\icnas4.cc.ic.ac.uk\fl4718\Desktop\Machine learning\Data\test.csv'

# read data from local address
train_df_full = pd.read_csv(train_path, index_col=0)
structures_df_full = pd.read_csv(structures_path, dtype={'atom_index': np.int8})
test_df_full = pd.read_csv(test_path)

# Add distance feature to the test and trin data
train_df_ = distance(train_df_full)
test_df_ = distance(test_df_full)

# ndarray with names of each atom in the structures csv
atom = structures_df_full['atom'].values

# Add electronegativity and radius colmun to the structures csv
structures_ = electronegativity(atom, structures_df_full)
structures_ = radius(atom, structures)

# Add number of bonds and connecting atoms columns
structures = n_bonds(structures)

# Add hybridization column
structures = hybridization(structures)

# Add pi_bonds column
structures = pi_bonds(structures)
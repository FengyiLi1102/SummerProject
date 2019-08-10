#%%
# libraries required
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn import metrics
from sklearn import linear_model
import seaborn as sns
import numpy as np
import pandas as pd
import time
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
                                  left_on=['molecule_name', f'atom_index_{index}'],
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

    print(10*'-' + '{}'.format(structures.shape[0]) + 10*'-')
    
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

    print(10*'*' + '{}'.format(df.shape[0]) + 10*'*')

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

    print(10*'*' + '{}'.format(df_copy.shape[0]) + 10*'*')

    return df_copy


def hybridization(structures):
    """
    :param: structures - structures data

    Goal: Calculate each hybridization in the structures data

    Return: structure data with hybridization column added
    """
    
    print('Calculate hybridization......')
    print(10*'-' + '{}'.format(structures.shape[0]) + 10*'-')
    
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

    print(10*'-' + '{}'.format(structures.shape[0]) + 10*'-')

    return structures


def pi_bonds(structures):
    """
    :param: structures - structures data

    Goal: Calculate the number of pi_bonds for each atom

    Return: structures with pi_bonds column added
    """

    print('Calculate pi bonds......')
    print(10*'-' + '{}'.format(structures.shape[0]) + 10*'-')

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

    print(10*'-' + '{}'.format(structures.shape[0]) + 10*'-')

    return structures


def electronegativity(atom_name, structures):
    """
    :param: atom_name - list or np.ndarray consisting of name of atoms
    :param: structures - structures data

    Goal: Assign an electrinegativity for each atom

    Return: structures with electrineativity column added
    """

    print(10*'-' + '{}'.format(structures.shape[0]) + 10*'-')

    electronegativity = {'H':2.2, 'C':2.55, 'N':3.04, 'O':3.44, 'F':3.98}
    en_ = [electronegativity[x] for x in tqdm(atom_name)]

    structures['EN'] = en_

    print(10*'-' + '{}'.format(structures.shape[0]) + 10*'-')

    return structures


def radius(atom_name, structures):
    """
    :param: atom_name - list or np.ndarray consisting of name of atoms
    :param: structures - structures data

    Goal: Assign an radius for each atom

    Return: structures with radius column added
    """

    print(10*'-' + '{}'.format(structures.shape[0]) + 10*'-')

    atomic_radius = {'H':0.38, 'C':0.77, 'N':0.75, 'O':0.73, 'F':0.71} # Without fudge factor

    fudge_factor = 0.05
    atomic_radius = {k:v + fudge_factor for k,v in atomic_radius.items()}
    rd_ = [atomic_radius[x] for x in atom_name]

    structures['RD'] = rd_

    print(10*'-' + '{}'.format(structures.shape[0]) + 10*'-')

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

    print('Create closest points......')
    print(10*'-' + '{}'.format(df_train.shape[0]) + 10*'-')

    df_temp=df_train.loc[: ,["molecule_name", "atom_index_0", "atom_index_1", "distance", "x_0", "y_0", "z_0", "x_1", "y_1", "z_1"]].copy()
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
    
    print(10*'-' + '{}'.format(df_train.shape[0]) + 10*'-')

    return df_train


def add_cos_features(df):
    """
    :param: df - dataframe containing necessary data for calculating the cosine value

    Goal: Calculating cosine value

    Return: dataframe with cosine data added
    """

    print('Add cosine features......{}'.format(df.shape[0]))

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

    print(10*'-' + '{}'.format(df.shape[0]) + 10*'-')

    return df


def more_features(df, df_):
    
    print(10*'-' + '{}'.format(df.shape[0]) + 10*'-')
    df['distance_mean'] = df.groupby('molecule_name')['distance'].transform('mean')
    df['distance_std'] = df.groupby('molecule_name')['distance'].transform('std')
    df['distance_min'] = df.groupby('molecule_name')['distance'].transform('min')
    df['distance_max'] = df.groupby('molecule_name')['distance'].transform('max')
    print(10*'-' + '{}'.format(df.shape[0]) + 10*'-')

    df['pi_bonds_mean'] = df_.groupby('molecule_name')['pi_bonds'].transform('mean')
    df['pi_bonds_std'] = df_.groupby('molecule_name')['pi_bonds'].transform('std')
    df['pi_bonds_min'] = df_.groupby('molecule_name')['pi_bonds'].transform('min')
    df['pi_bonds_max'] = df_.groupby('molecule_name')['pi_bonds'].transform('max')
    print(10*'-' + '{}'.format(df.shape[0]) + 10*'-')

    df['hybri_mean'] = df_.groupby('molecule_name')['hybri'].transform('mean')
    df['hybri_std'] = df_.groupby('molecule_name')['hybri'].transform('std')
    df['hybri_min'] = df_.groupby('molecule_name')['hybri'].transform('min')
    df['hybri_max'] = df_.groupby('molecule_name')['hybri'].transform('max')
    print(10*'-' + '{}'.format(df.shape[0]) + 10*'-')

    df['EN_mean'] = df['EN_0'] + df['EN_1'] / 2
    df['RD_mean'] = df['RD_0'] + df['RD_1'] / 2
    print(10*'-' + '{}'.format(df.shape[0]) + 10*'-')

    print('Add mean, std, min and max of bond lengths for atom 0......')

    df['bond_length_0_mean'] = [np.mean(df.loc[i, 'bond_lengths_0']) for i in tqdm(range(len(df.index)))]
    df['bond_length_0_std'] = [np.std(df.loc[i, 'bond_lengths_0']) for i in tqdm(range(len(df.index)))]
    df['bond_length_0_min'] = [np.min(df.loc[i, 'bond_lengths_0']) for i in tqdm(range(len(df.index)))]
    df['bond_length_0_max'] = [np.max(df.loc[i, 'bond_lengths_0']) for i in tqdm(range(len(df.index)))]
    print(10*'-' + '{}'.format(df.shape[0]) + 10*'-')

    print('Add mean,std, min and max of bond lengths for atom 1......')

    df['bond_length_1_mean'] = [np.mean(df.loc[i, 'bond_lengths_1']) for i in tqdm(range(len(df.index)))]
    df['bond_length_1_std'] = [np.std(df.loc[i, 'bond_lengths_1']) for i in tqdm(range(len(df.index)))]
    df['bond_length_1_min'] = [np.min(df.loc[i, 'bond_lengths_1']) for i in tqdm(range(len(df.index)))]
    df['bond_length_1_max'] = [np.max(df.loc[i, 'bond_lengths_1']) for i in tqdm(range(len(df.index)))]
    print(10*'-' + '{}'.format(df.shape[0]) + 10*'-')

    return df


def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true-y_pred).abs().groupby(types).mean()

    return np.log(maes.map(lambda x: max(x, floor))).mean()


def train_model_regression(X, X_test, y, params, folds, model_type='lgb', eval_metric='mae', columns=None, plot_feature_importance=False, model=None,
                           verbose=10000, early_stopping_rounds=200, n_estimators=50000):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.
    
    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type
    
    """
    columns = X.columns if columns is None else columns
    X_test = X_test[list(columns)]
    
    # to set up scoring parameters
    metrics_dict = {'mae': {'lgb_metric_name': 'mae',
                        'catboost_metric_name': 'MAE',
                        'sklearn_scoring_function': metrics.mean_absolute_error},
                    'group_mae': {'lgb_metric_name': 'mae',
                        'catboost_metric_name': 'MAE',
                        'scoring_function': group_mean_log_mae},
                    'mse': {'lgb_metric_name': 'mse',
                        'catboost_metric_name': 'MSE',
                        'sklearn_scoring_function': metrics.mean_squared_error}
                    }

    result_dict = {}
    
    # out-of-fold predictions on train data
    oof = np.zeros(len(X))
    
    # averaged predictions on train data
    prediction = np.zeros(len(X_test))
    
    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()
    
    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators = n_estimators, n_jobs = -1)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=verbose, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
        
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')
            
            y_pred = model.predict(X_test).reshape(-1,)
        
        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000,  eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,
                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)
        
        oof[valid_index] = y_pred_valid.reshape(-1,)
        if eval_metric != 'group_mae':
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
        else:
            scores.append(metrics_dict[eval_metric]['scoring_function'](y_valid, y_pred_valid, X_valid['type']))

        prediction += y_pred    
        
        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= folds.n_splits
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores
    
    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= folds.n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12))
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
            plt.title('LGB Features (avg over folds)')
            
            result_dict['feature_importance'] = feature_importance
        
    return result_dict


def data_generation(df, columns, draft, file_name):

    # Select columns for data
    df_plus = df[columns]

    # Store data for further using
    df_plus.to_csv(r'G:\{}_d{}.csv'.format(file_name, draft), index=False)

    print(10*'-' +'Done!' + 10*'-')

    return None


def structures_prepa(structures_df_full):

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

    structures.to_csv(r'G:\structures.csv')

    return structures


def data_prepa(df1, structures_df_full, structures):

    # Add distance feature to the test and trin data
    df = distance(df1, structures_df_full)

    # Merge structures data and train data
    struc_df = struc1_merge(df, structures, 0)
    struc_df = struc1_merge(struc_df, structures, 1)

    struc_df = struc_df.drop(['atom_index_x', 'atom_x', 'x_x', 'y_x', 'z_x',
                              'atom_index_y', 'atom_y','x_y', 'y_y', 'z_y'], axis=1
                            )

    # Add bond angle column
    struc_df = create_closest(struc_df)
    struc_df = add_cos_features(struc_df)

    # Add more features
    struc_df = more_features(struc_df, structures)

    # Missing data in columns with std values
    struc_df['Angle'] = struc_df['Angle'].fillna(180.0)
    struc_df['distance_std'] = struc_df['distance_std'].fillna(0.0)

    return struc_df


def model_train_set(train, test, params, model_type, fold_n):

    # Set parameters for model training
    folds = KFold(n_splits=fold_n, shuffle=False, random_state=0)

    test_submission = pd.DataFrame(columns=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])
    
    # Train the model by each type
    types = test.type.unique()

    results_all = dict()
    test_copy = test_df_full.copy()
    
    for i in types:
        train_plus = train[train['type'] == i]
        test_plus = test[test['type'] == i]
        y = train_plus['scalar_coupling_constant']
        train_plus = train_plus.drop(['type', 'molecule_name', 'scalar_coupling_constant'], axis=1)
        test_plus = test_plus.drop(['type', 'molecule_name'], axis=1)

        print('\n' + 15*'-' + 'TYPE {}'.format(i) + 15*'-' + '\n')

        if i[0] == '1':
            train_plus = train_plus.drop(['Angle'], axis=1)
            results = train_model_regression(train_plus, test_plus, y, params, folds, model_type=model_type)
            results_all[i] = results
        else:
            results = train_model_regression(train_plus, test_plus, y, params, folds, model_type=model_type)
            results_all[i] = results
        
        test_plus_copy = test_copy[test_copy['type'] == i]
        test_plus_copy['prediction'] = results['prediction']
        test_submission = pd.concat([test_submission, test_plus_copy], ignore_index=False)

    submission['scalar_coupling_constant'] = test_submission['prediction'].values
    submission.to_csv(r'G:\submission_d2.csv', index=False)
    
    return results_all


# File paths
#train_path = r'\\icnas4.cc.ic.ac.uk\fl4718\Desktop\Machine learning\Data\train.csv'
structures_path = r'\\icnas4.cc.ic.ac.uk\fl4718\Desktop\Machine learning\Data\structures.csv'
#test_path = r'\\icnas4.cc.ic.ac.uk\fl4718\Desktop\Machine learning\Data\test.csv'
submission_path = r'\\icnas4.cc.ic.ac.uk\fl4718\Desktop\Machine learning\Data\sample_submission.csv'
qm9_train_path = r'G:\qm9_train.csv'
qm9_test_path = r'G:\qm9_test.csv'

# read data from local address
#train_df_full = pd.read_csv(train_path, index_col=0, dtype={'atom_index_0': np.int8, 'atom_index_1': np.int8})
structures_df_full = pd.read_csv(structures_path, dtype={'atom_index': np.int8})
#test_df_full = pd.read_csv(test_path, index_col=0, dtype={'atom_index_0': np.int8, 'atom_index_1': np.int8})
submission = pd.read_csv(submission_path)
qm9_train = pd.read_csv(qm9_train_path)
qm9_test = pd.read_csv(qm9_test_path)

# Structures preparation
structures = pd.read_csv(r'G:\structures.csv')
#%%
# Data preparations for training
struc_train = data_prepa(train_df_full, structures_df_full, structures)
struc_test = data_prepa(test_df_full, structures_df_full, structures)

# Columns of features
good_columns_train = [  'molecule_name',
                        'type',
                        'scalar_coupling_constant',
                        'distance',
                        'EN_0',
                        'RD_0',
                        'hybri_0',
                        'pi_bonds_0',
                        'EN_1',
                        'RD_1',
                        'EN_mean',
                        'RD_mean',
                        'hybri_1',
                        'pi_bonds_1',
                        'Angle',
                        'distance_mean',
                        'distance_std',
                        'hybri_mean',
                        'hybri_std',
                        'hybri_min',
                        'hybri_max',
                        'pi_bonds_min',
                        'pi_bonds_max',
                        'pi_bonds_mean',
                        'pi_bonds_std',
                        'bond_length_1_mean',
                        'bond_length_1_std',
                        'bond_length_1_min',
                        'bond_length_1_max',
                        'bond_length_0_min',
                        'bond_length_0_max',
                        'bond_length_0_mean',
                        'bond_length_0_std',
                        'x_0',
                        'y_0',
                        'z_0',
                        'x_1',
                        'y_1',
                        'z_1'
                     ]
# Data generation for training data
data_generation(struc_train, good_columns_train, '2', 'train')

# Data generation for test data
good_columns_train.remove('scalar_coupling_constant')
good_columns_test = good_columns_train
data_generation(struc_test, good_columns_test, '2', 'test')

# Import prepared data
train_d2 = pd.read_csv(r'G:\train_d2.csv')
test_d2 = pd.read_csv(r'G:\test_d2.csv')

params_grid = { 'num_leaves': [50, 60, 70],
                'min_child_samples': [79, 89, 99],
                'min_data_in_leaf' : [100, 200, 300],
                'objective': ['regression'],
                'max_depth': [9, 15, 20],
                'learning_rate': [0.05, 0.1, 0.2],
                "boosting_type": ["gbdt"],
                "subsample_freq": [1],
                "subsample": [0.9],
                "bagging_seed": [11],
                "metric": ['mae'],
                "verbosity": [-1],
                'reg_alpha': [0.1],
                'reg_lambda': [0.3],
                'colsample_bytree': [1.0]
                }

params_lgb = {'num_leaves': 10,
          'min_child_samples': 79,
          'min_data_in_leaf' : 100,
          'objective': 'regression',
          'max_depth': 9,
          'learning_rate': 0.2,
          "boosting_type": "gbdt",
          "subsample_freq": 1,
          "subsample": 0.9,
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
          'reg_alpha': 0.1,
          'reg_lambda': 0.3,
          'colsample_bytree': 1.0
         }

params_xgb = {'booster': 'gbtree',
              'verbosity': 1,
              'eta': 0.3,
              'gamma': 3,
              'max_depth': 12,
              'min_child_weight': 1,
              'subsample': 0.5,
              'lambda': 1,
              'alpha': 0
              }

model_train_set(train_d2, test_d2, params_lgb, 'lgb', 4)

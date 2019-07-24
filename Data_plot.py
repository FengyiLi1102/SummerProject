import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def score_mae(X_train, X_valid, y_train, y_valid):
    '''
    Variables:
    X_train: the training data
    X_valid: the data for testing
    y_train: target value for training
    y_valid: target data for validation

    Function:
    The model chosen is the random forest and the random state is 0. The model is trained by four variables and predicted target
    values are given. The mean absolute error is calculated between the validated and predicted data. 
    '''

    my_model = RandomForestRegressor(random_state=0)
    my_model.fit(X_train, y_train)
    value_predict = my_model.predict(X_valid)

    return mean_absolute_error(y_valid, value_predict)

# read the data file train.csv
train = pd.read_csv(r'\\icnas4.cc.ic.ac.uk\fl4718\Desktop\Machine learning\Data\train.csv', index_col=0)

# Drop the target column scalar_coupling_constant and molecule_name (easy for model processing). Choose the target variable.
X = train.drop(['scalar_coupling_constant', 'molecule_name'], axis=1)
y = train.scalar_coupling_constant

# Split the data into training and validation parts.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

# Create two copies of X dataframe in order not to change the original data
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Use LabelEncoder to replace the nominal catergorical varible 'type' into integers for processing
label_encoder = LabelEncoder()
X_train_plus.type = label_encoder.fit_transform(X_train.type)
X_valid_plus.type = label_encoder.transform(X_valid.type)

# print the mean absolute error for the training results.
print('The mean absolute error for this model is: {}'.format(score_mae(X_train_plus, X_valid_plus, y_train, y_valid)))


'''
Thinking: For some scc, the values are quite small, which can be seen from the head of the train.csv table. The mae value is 
          still too large compared with them. It is obivous that my trial is not worth as a mertual model for further improvement.
          The mae for some large scc atoms shows a dependance between type and scc. Also, I dropped the molecular_name column that
          should be considered for easy processing. Same type of atoms at the same position in different molecules may have different
          scc values, which can cause errors in the model. Scientific background is also required for us to make sure type is related
          with the scc. Other variables will definitely improve our final results and we need to explore the relationships between them
          and scc.

          I will try to finish Intermediate ML and Data Visualisation as soon as possible and make some visuable plots of our data.
'''
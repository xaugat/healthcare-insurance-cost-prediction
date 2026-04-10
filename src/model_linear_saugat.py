import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/insurance.csv')

# Display the first few rows of the DataFrame
print(df.head())
# Display summary statistics of the DataFrame
print(df.describe())
# Display the data types of each column
print(df.columns.tolist())

#preprocessing
#encoding categorical variables
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

#creating log-transformed target variable
df_encoded['log_charges'] = np.log(df_encoded['charges'])

#features and target variable
feature_columns = [col for col in df_encoded.columns if col not in ['charges', 'log_charges']]

X = df_encoded[feature_columns].astype(float).values
y_original = df_encoded['charges'].values.reshape(-1, 1)
y_log = df_encoded['log_charges'].values.reshape(-1, 1)

#train-test split dataset

def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    test_count = int(len(indices) * test_size)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

#slit original target variable
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y_original)
#split log-transformed target variable
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X, y_log)
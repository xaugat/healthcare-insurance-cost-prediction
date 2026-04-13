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

#feature scaling
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def initialize_parameters(self, n_features):
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

    def compute_cost(self, X, y):
        m = X.shape[0]
        y_pred = np.dot(X, self.weights) + self.bias
        cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
        return cost
    
    def fit(self, X, y):
        m,n = X.shape
        self.initialize_parameters(n)

        for i in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

            cost = self.compute_cost(X, y)
            self.loss_history.append(cost)

            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost}")

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred





    

    

    
    
    

    



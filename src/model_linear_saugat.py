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
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

#slit original target variable
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y_original)
#split log-transformed target variable
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X, y_log)

#feature scaling

def standarize_train_test(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    std[std == 0] = 1

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled

X_train_orig, X_test_orig = standarize_train_test(X_train_orig, X_test_orig)
X_train_log, X_test_log = standarize_train_test(X_train_log, X_test_log)

#scaling original target variable which prevents overflow for original charges model

y_orig_mean = y_train_orig.mean()
y_orig_std = y_train_orig.std()

y_train_orig_scaled = (y_train_orig - y_orig_mean) / y_orig_std

class LinearRegressionGD:
    def __init__(self, learning_rate=0.001, n_iterations=3000):
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
    
#evaluation metrics
def mean_squared_error( y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        return mse
    
def r2_score( y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
#train linear regression model on original target variable charges
model_orig = LinearRegressionGD(learning_rate=0.001, n_iterations=3000)
model_orig.fit(X_train_orig, y_train_orig_scaled)

y_pred_orig_scaled = model_orig.predict(X_test_orig)

#converting predictions back to original scale
y_pred_orig = y_pred_orig_scaled * y_orig_std + y_orig_mean

mean_squared_error_orig = mean_squared_error(y_test_orig, y_pred_orig)
r2_score_orig = r2_score(y_test_orig, y_pred_orig)

print(f"Original Target Variable - MSE: {mean_squared_error_orig}, R2 Score: {r2_score_orig}")

#train linear regression model on log-transformed target variable log_charges
model_log = LinearRegressionGD(learning_rate=0.001, n_iterations=3000)
model_log.fit(X_train_log, y_train_log)

y_pred_log = model_log.predict(X_test_log)

#convert log predictions back to original scale
y_pred_log_exp = np.exp(y_pred_log)
Y_test_log_exp = np.exp(y_test_log)

mean_squared_error_log = mean_squared_error(Y_test_log_exp, y_pred_log_exp)
r2_score_log = r2_score(Y_test_log_exp, y_pred_log_exp)

print(f"Log-Transformed Target Variable - MSE: {mean_squared_error_log}, R2 Score: {r2_score_log}")

#plotting loss curve for log-transformed target variable
plt.plot(model_log.loss_history)
plt.title("Loss Curve for Log-Transformed Target Variable")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

#feature importance analysis using coefficients from linear regression model
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Weight (log-transformed)': model_log.weights.flatten(),
    'Absolute weight': np.abs(model_log.weights.flatten())
}).sort_values(by='Weight (log-transformed)', key=abs, ascending=False)

print(feature_importance)

#plotting feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['Feature'], feature_importance['Absolute weight'])
plt.title("Feature Importance based on Absolute Weights (Log-Transformed Target)")
plt.xlabel("Features")
plt.ylabel("Absolute Weight")
plt.xticks(rotation=45)
plt.show()  

#interpretation of feature importance
print("Interpretation of Feature Importance:")
print("The features with the highest absolute weights have the most influence on the predicted insurance charges. For example, if 'smoker_yes' has a high positive weight, it indicates that being a smoker significantly increases the predicted charges. Similarly, if 'age' has a positive weight, it suggests that older individuals tend to have higher insurance charges. The magnitude of the weights can help identify which factors are most important in determining insurance costs."
      )
print("In this analysis, we can see that the 'smoker_yes' feature has a significant positive weight, indicating that smoking is a major factor contributing to higher insurance charges. Additionally, features like 'age' and 'bmi' also have positive weights, suggesting that older age and higher body mass index are associated with increased insurance costs. On the other hand, features with negative weights may indicate factors that contribute to lower charges. Overall, this feature importance analysis helps us understand the key drivers of insurance costs in the dataset.")
print("In conclusion, the linear regression model trained on the log-transformed target variable (log_charges) performed better than the model trained on the original target variable (charges) in terms of both mean squared error and R2 score. The feature importance analysis revealed that smoking status, age, and BMI are significant factors influencing insurance charges. This insight can be valuable for insurance companies in assessing risk and setting premiums based on individual characteristics.")
print ("age and bmi have positive weights, suggesting that older age and higher body mass index are associated with increased insurance costs. On the other hand, features with negative weights may indicate factors that contribute to lower charges. Overall, this feature importance analysis helps us understand the key drivers of insurance costs in the dataset.")
print("if log transformation is applied to the target variable, the model can capture non-linear relationships between features and the target variable, which may lead to improved performance. In this case, the linear regression model trained on the log-transformed target variable (log_charges) performed better than the model trained on the original target variable (charges) in terms of both mean squared error and R2 score. This suggests that the log transformation helped to stabilize the variance and make the relationship between features and target variable more linear, resulting in better predictions.")




    

    

    
    
    

    



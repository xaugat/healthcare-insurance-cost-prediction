"""
comparative_analysis.py

This file compares performance of following 
on same dataset:
- Linear regression
- Decision tree
- Random Forest
- Sklearn random forest

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from model_rf_rshijuta import build_tree, predict_tree, random_forest, predict_forest
from model_linear_saugat import LinearRegressionGD

df = pd.read_csv("data/raw/insurance.csv")
df = pd.get_dummies(df, drop_first=True)

X = df.drop("charges", axis=1).astype(float).values
y=df["charges"].values.reshape(-1,1)
y_log=np.log(y)

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)
_, _, y_train_log, y_test_log=train_test_split(X,y_log,test_size=0.2,random_state=42)


def standardize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1
    return (X_train - mean)/std, (X_test - mean)/std

X_train_scaled, X_test_scaled = standardize(X_train, X_test)

# Metrics for evaluation
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def r2_score(y_true, y_pred):
    ss_total =np.sum((y_true-np.mean(y_true)) **2)
    ss_res = np.sum((y_true-y_pred)** 2)
    return 1 - (ss_res/ss_total)

# Linear regression
lin_model = LinearRegressionGD(learning_rate=0.001,n_iterations=3000)
lin_model.fit(X_train_scaled,y_train)
lin_preds = lin_model.predict(X_test_scaled)

# log version
lin_log_model=LinearRegressionGD(learning_rate=0.001,n_iterations=3000)
lin_log_model.fit(X_train_scaled,y_train_log)
lin_log_preds = np.exp(lin_log_model.predict(X_test_scaled))

# Decision tree
tree=build_tree(X_train,y_train.flatten(),max_depth=7)
tree_preds=np.array([predict_tree(tree,x) for x in X_test]).reshape(-1,1)

# random forest implementations (scratch)
forest =random_forest(X_train, y_train.flatten(),n_trees=50,max_depth=7)
rf_preds =predict_forest(forest,X_test).reshape(-1,1)

# sklearn Rf
sk_model=RandomForestRegressor()
sk_model.fit(X_train,y_train.ravel())
sk_preds = sk_model.predict(X_test).reshape(-1,1)

# COMPARISOM TABLE
results = []


def evaluate(name, y_true, y_pred):
    results.append({
        "Model":name,
        "MAE":mae(y_true,y_pred),
        "MSE":mse(y_true, y_pred),
        "RMSE":rmse(y_true, y_pred),
        "R2":r2_score(y_true, y_pred)
    })

evaluate("Linear Regression", y_test,lin_preds)
evaluate("Linear Regression(Log)",y_test,lin_log_preds)
evaluate("Decision Tree", y_test,tree_preds)
evaluate("Random Forest(Scratch)",y_test,rf_preds)
evaluate("Random Forest(Sklearn)",y_test,sk_preds)

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON\n")
print(results_df.sort_values(by="R2",ascending=False))

results_df.set_index("Model")[["MAE","RMSE"]].plot(kind="bar")
plt.title("Model Comparison")
plt.xticks(rotation=45)
plt.show()




"""
evaluate_rf.py

Here we evaluate Decision Tree and Random Forest Models 
using Mean absolute error on our test data 
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from model_rf_rshijuta import(
    build_tree, predict_tree, random_forest, predict_forest)

df = pd.read_csv("data/raw/insurance.csv")
df = pd.get_dummies(df, drop_first=True)

# split features and target
X = df.drop("charges", axis=1).values
y= df["charges"].values

#train - test split (80-20)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# train models:

tree = build_tree(X_train, y_train, max_depth=7)
forest= random_forest(X_train, y_train, n_trees=50, max_depth=7)

"""
Predict values:
Decision tree - 1 pred.  per sample
Random Forest - average of multiple trees
"""

tree_preds = [predict_tree(tree,x) for x in X_test]
rf_preds = predict_forest(forest, X_test)

# Mean absolute error
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# mean square error
def mse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)

# root mean square error
def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

# R-square score
def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual= np.sum((y_true - y_pred)** 2)
    return 1 - (ss_residual/ss_total)


tree_mae = mae(y_test, tree_preds)
rf_mae = mae(y_test, rf_preds)

tree_mse = mse(y_test, tree_preds)
rf_mse = mse(y_test, rf_preds)

tree_rmse = rmse(y_test, tree_preds)
rf_rmse = rmse(y_test, rf_preds)

tree_r2 = r2_score(y_test,tree_preds)
rf_r2 = r2_score(y_test, rf_preds)

print("\nDecision tree:")
print(f"MAE:{tree_mae:.2f}")
print(f"MSE:{tree_mse:.2f}")
print(f"RMSE:{tree_rmse:.2f}")
print(f"R2:{tree_r2:.2f}")

print("\nRandom forest:")
print(f"MAE:{rf_mae:.2f}")
print(f"MSE:{rf_mse:.2f}")
print(f"RMSE:{rf_rmse:.2f}")
print(f"R2:{rf_r2:.2f}")


"""
Compare with sklearn
"""
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)

sk_preds = model.predict(X_test)

print ("Sklearn RF MAE:", mae(y_test, sk_preds))

"""
(04/12/2026)
Output: 
Decision tree Mean abs. error: 2685.244901087169
Random forest Mean abs. error: 2598.9515725650263
Sklearn RF MAE: 2503.348150366294

Takeaways
- RF performed better that decision tree with lower mean abs. error
- ensemle learning improved prediction
- our results were close to sklearn's model

"""

"""
(4/13/2026)
Added mse, rmse and r2 score  metrics

OUTPUT

Decision tree:

MAE:2685.24
MSE:21333531.82
RMSE:4618.82
R2:0.86

Random forest:

MAE:2562.54
MSE:20836196.20
RMSE:4564.67
R2:0.87

Sklearn RF MAE: 2529.5996100911075
"""
"""
explainability_rf.py

This file is to explain the predictions of RF model
using SHAP

Although we built RF model from scratch to understand how the algorithm works, 
we have used sklearn's RandomForestRegressor for SHAP explainability.

(SHAP requires a model that is fully compatible with internal 
TreeExplainer methods)

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/insurance.csv")
df = pd.get_dummies(df, drop_first=True)

X= df.drop("charges", axis=1)
y= df["charges"]

feature_names = X.columns

#train - test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# train sklearn random forest 
model = RandomForestRegressor()
model.fit(X_train,y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# global feature importance
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Local explanation : single prediction
shap.plots.waterfall(shap.Explanation(
    values= shap_values[0], 
    base_values=explainer.expected_value,
    data= X_test.iloc[0],
    feature_names=feature_names))

plt.show()


"""

Output Interpretations: 

SHAP summary plot shows: 
'smoker', 'age' and 'bmi' have highest impact on charges 
compared to 'sex' and region.

Waterfall plot: 
this explains single prediction.
it shows how features contribute to final output

"""

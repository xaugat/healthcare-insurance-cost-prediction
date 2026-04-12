#model_rf_rshijuta.py
import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/insurance.csv")
print(df.head())

# converting categorical values like sex, smoker, region into numbers
df= pd.get_dummies(df, drop_first=True)


# X : input features
# y: target
X = df.drop("charges", axis=1)
y= df["charges"]
X = X.values
y = y.values

def variance(y):
    return np.var(y)

# splitting dataset into 2 groups basing on threshold
def split_dataset(X, y, feature, threshold):
    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold

    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]


"""
To find the best way to divide dataset into two groups:
First. it loops over every feature and threshold values
For each split, data is dividesd into left and right parts.
We use "weighted variance" to calculate the weight of split (good or baD)
Goal ==> minimize the variance
Returns: feature anf threshold that gives min_variance
"""

def best_split(X, y):
    best_feature = None
    best_threshold = None
    best_score = float("inf")

    n_features = X.shape[1]

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])

        for t in thresholds:
            X_l, X_r, y_l, y_r = split_dataset(X, y, feature, t)

            if len(y_l) == 0 or len(y_r) == 0:
                continue

            score = (len(y_l)/len(y)) * variance(y_l) + (len(y_r)/len(y)) * variance(y_r)

            if score < best_score:
                best_score = score
                best_feature = feature
                best_threshold = t
    
    return best_feature, best_threshold


"""
Recursively construct decision tree:
It checks at each step if max_depth or low_variance is reached.
If yes, it returns mean value of leaf node (i.e prediction)
Else, it finds best feature and threshold to split data.
It divides data into left and right subset, the function calls itself to build subtrees.
Returns: dictionary of current node and its children

"""
def build_tree(X, y, depth=0, max_depth=3):
    if depth == max_depth or np.var(y)==0:
        return np.mean(y)
    
    feature, threshold = best_split(X, y)

    if feature is None: 
        return np.mean(y)

    X_l, X_r, y_l, y_r = split_dataset(X, y, feature, threshold)

    return { "feature": feature, "threshold": threshold,
            "left": build_tree(X_l, y_l, depth+1, max_depth),
            "right": build_tree(X_r, y_r, depth+1, max_depth)}


"""
to make prediction for a single data point using trained tree:
starts at root node, checks the feature and threshold condition
moves acc. to the condition, recursively until it reaches leaf node
"""
def predict_tree(tree,x):
    if not isinstance(tree,dict):
        return tree
    
    if x[tree["feature"]] <= tree["threshold"]:
        return predict_tree(tree["left"], x)
    else:
        return predict_tree(tree["right"], x)
    
tree= build_tree(X, y, max_depth=3)
pred=predict_tree(tree, X[0])

print("Prediction:", pred)
print("Actual:", y[0])


"""
Output:
   age     sex  ...     region      charges
0   19  female  ...  southwest  16884.92400
1   18    male  ...  southeast   1725.55230
2   28    male  ...  southeast   4449.46200
3   33    male  ...  northwest  21984.47061
4   32    male  ...  northwest   3866.85520

[5 rows x 7 columns]
Prediction: 18474.021906575344
Actual: 16884.924


Interpretation:
The prediction is close to actual but not exact: 

"""
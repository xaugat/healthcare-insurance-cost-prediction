#model_rf_rshijuta.py
import pandas as pd
import numpy as np

df = pd.read_csv("data/insurance.csv")
#print(df.head())

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
        thresholds = np.unique(X[:, feature])[:20]

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
def build_tree(X, y, depth=0, max_depth=7):
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



"""
Random Forest: 
Now, we terain multiple trees on different random samples of data
(bootstrappping)
Final prediction = average prediction from all trees
This is supposed to improve accuracy and reduce overfitting
"""

def bootstrap_sample(X,y):
    n_samples=X.shape[0]
    indices = np.random.choice(n_samples,n_samples,replace=True)
    return X[indices], y[indices]

def random_forest(X, y, n_trees=50, max_depth=7):
    trees=[]

    for i in range(n_trees):
        X_sample, y_sample = bootstrap_sample(X,y)
        tree= build_tree(X_sample, y_sample, max_depth=max_depth)
        trees.append(tree)

    return trees

def predict_forest(trees, X):
    predictions=[]

    for tree in trees: 
        preds = [predict_tree(tree,x) for x in X]
        predictions.append(preds)

    return np.mean(predictions, axis=0)

# decision tree & rf model test

if __name__ == "__main__": 
    tree= build_tree(X, y, max_depth=7)
    pred=predict_tree(tree, X[1])

    print("Prediction:", pred)
    print("Actual:", y[1])

    forest = random_forest(X,y,n_trees=50, max_depth=7)
    preds = predict_forest(forest,X)

    print("Random Forest prediction:", preds[1])
    print("Actual:", y[1])

"""
(04/12/2026) Improvements/ changes/ addition: 
 
- Increased tree depth & no, of trees
- Implemented random forest
- limited threshold 
"""
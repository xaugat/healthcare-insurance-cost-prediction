# healthcare-insurance-cost-prediction

This project focuses on predicting healthcare insurance costs using different machine learning models and comparing their performances. 

## Models Used
- Linear Regression (implemented from scratch)
- Decision Tree (implemented from scratch)
- Random Forest (implemented from scratch)
- Random Forest (scikit-learn)

## How to run:

pip install -r requirements.txt
python src/comparative_analysis.py
python src/explainability_rf.py

## Dataset
File : data/raw/insurance.csv
or data/insurance.csv

## File Structure and Description
src/data_exploration.ipynb
Performs initial data analysis and visualisation

src/model_linear_saugat.py
Implements Linear Regression using Gradient Descent

src/model_rf_rshijuta.py
Implements Decision Tree anf Random forest from scratch

src/evaluate_rf.py
Evaluates decision tree and random forest models separately

src/explainability_rf.py
Performs SHAP analysis for model interpretability


src/comparative_analysis.py
Compares all models and generates evaluation metrics and plots

## Authors
Rshijuta Pokharel: Data exploration, Random Forest implementation, SHAP analysis, comparative analysis, Report writing

Saugat Poudel: Linear Regression implementation, Experimental setup, Visualisations, comparative analysis, Report writing


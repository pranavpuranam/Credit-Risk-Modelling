"""
Filename: testing.py

In this section of code we'll test the ML models on the test.csv dataset.

"""

# necessary imports

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# read in the testing file
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# now we define our target X and y

X = train.drop('loan_status', axis = 1)

y = train['loan_status']

# define a function that can evaluate the performance of a model

def evaluate_model(model_func, X, y):
    model = model_func(X,y)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y,y_pred)
"""
Filename: training.py

In this section of code we'll build ML models and train them on the train.csv dataset.

"""

# necessary imports

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# define the logistic regression model

def logistic_regression(X_train,y_train):
    
    # define the logistic regression model

    model = LogisticRegression()

    model.fit(X_train,y_train)

    return model
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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from training import logistic_regression

# read in the testing file
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 

# now we define our Xs and ys

X_test = test.drop('loan_status', axis=1)
X_train = train.drop('loan_status',axis=1)

y_test = test['loan_status']
y_train = train['loan_status'] 

# define a function that can evaluate the performance of a model

def evaluate_model(model_func, X_train, y_train, X_test, y_test, model_name):
    
    # name the model which is being evaluated

    print(f"\nEvaluating {model_name}...")

    # run model on the training data

    model = model_func(X_train, y_train)

    # apply model to the testing data

    y_pred = model.predict(X_test)

    # output the reliability scores

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# evaluate the logistic regression model

evaluate_model(logistic_regression, X_train, y_train, X_test, y_test, "Logistic Regression")


"""
Filename: preprocessing.py

In this section of code we'll look at the credit_risk_dataset.csv file and separate it into two files: one for training and one for backtesting.

The backtesting file will be assembled from every 10th row in the original file to avoid bias in the backtesting data.

"""

# necessary imports

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# read in the whole file

raw = pd.read_csv("credit_risk_dataset.csv")

# remove all the rows with incomplete info

raw = raw.dropna()

# transform the loan_grade column into numerical

label_encoder = LabelEncoder()
raw['loan_grade'] = label_encoder.fit_transform(raw['loan_grade'])

# transform the cb_person_default_on_file column into numerical
raw['cb_person_default_on_file'] = raw['cb_person_default_on_file'].map({'Y': True, 'N': False})

# transform the person_home_ownership column into numerical

raw = pd.get_dummies(raw, columns=['person_home_ownership'], prefix='Ownership Status: ')

# transform the loan_intent into numerical

raw = pd.get_dummies(raw, columns = ['loan_intent'], prefix = "Intent of Loan: ")

# find out more about the characteristics of the dataset

print(raw.head())

print(raw.info())

# raw.to_csv('num-data.csv', index = False)



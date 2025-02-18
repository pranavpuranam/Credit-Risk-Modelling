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

# read in the whole file

raw = pd.read_csv("credit_risk_dataset.csv")

# remove all the rows with incomplete info

raw = raw.dropna()

# find out more about the characteristics of the dataset

print(raw.head())

print(raw.info())

# split data into training and testing sets (80% for training and 20% for testing)

train_df, test_df = train_test_split(raw, test_size=0.2, random_state=42)

# save all to csv file format

# train_df.to_csv('train.csv', index=False)
# test_df.to_csv('test.csv', index=False)




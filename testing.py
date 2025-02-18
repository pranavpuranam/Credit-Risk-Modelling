"""
Filename: testing.py

In this section of code we'll test the ML models on the test.csv dataset.

"""

# necessary imports

import matplotlib as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from training import logistic_regression

# read in the testing file
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# now we define our Xs and ys

X_test = test.drop('loan_status', axis=1)
X_train = train.drop('loan_status',axis=1)

y_test = test['loan_status']
y_train = train['loan_status'] 

# define a function that can evaluate the performance of a model

def evaluate_model(model_func, X_train, y_train, X_test, y_test, model_name):

    """
    Trains and evaluates a model, displaying key metrics and visualizations.

    Parameters:
        model_func (function): A function that initializes and trains the model
        X_train (DataFrame): Training features
        y_train (Series): Training target
        X_test (DataFrame): Testing features
        y_test (Series): Testing target
        model_name (str): Name of the model being evaluated

    Returns:
        None
    """

    # Train the model
    model = model_func(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]  # Get probability scores for ROC and PR curves

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Create some visualisations for the accuracy of the ML model

    # === Confusion Matrix Heatmap ===
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Default", "Default"], yticklabels=["No Default", "Default"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

    """
    # === ROC Curve ===
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random model baseline
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.show()

    # === Precision-Recall Curve ===
    precision, recall, _ = precision_recall_curve(y_test, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='red', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.show()

    # === Feature Importance (if applicable) ===
    if hasattr(model, 'coef_'):  # Only for models with coefficients (e.g., Logistic Regression)
        feature_importance = np.abs(model.coef_[0])  # Get absolute coefficient values
        features = X_train.columns

        plt.figure(figsize=(8, 6))
        sns.barplot(x=feature_importance, y=features, color='green')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance - {model_name}')
        plt.show()
    
    """

# evaluate the logistic regression model

evaluate_model(logistic_regression, X_train, y_train, X_test, y_test, "Logistic Regression")


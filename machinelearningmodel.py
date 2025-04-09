# -*- coding: utf-8 -*-
"""
Modified version of: MachineLearningModel.ipynb
Original source: https://colab.research.google.com/drive/1uHtmkAXO5elVax3lvqnX4j9WUxSX2FWN

This version includes improved comments, clearer variable names, basic modularization, and logging support.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load dataset
logging.info("Loading dataset...")
data_url = 'https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv'
dataset = pd.read_csv(data_url)

# Separate features and target
logging.info("Separating features and target variable...")
target = dataset['logS']
features = dataset.drop('logS', axis=1)

# Split data into training and testing sets
logging.info("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=100
)

# Function to train a linear regression model
def train_linear_regression(X_train, y_train):
    logging.info("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Train the model
linear_model = train_linear_regression(X_train, y_train)

# Make predictions
logging.info("Making predictions...")
train_predictions = linear_model.predict(X_train)
test_predictions = linear_model.predict(X_test)

# Output preview
logging.info("Training and prediction completed successfully.")

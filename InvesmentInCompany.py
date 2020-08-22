# -*- coding: utf-8 -*-
#Multiple Linerar Regression used to find components that are best suited for invesment in order to get maximum profit.
"""
Created on Fri Aug 14 15:43:41 2020

@author: Moomina
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Converting Categorial Variable into numbers
from sklearn.preprocessing   import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("states", OneHotEncoder(),[3])], remainder="passthrough") # The last arg ([0]) is the list of columns you want to transform in this step
X=ct.fit_transform(X)    
#Avoiding the dummy variable trap
X= X[:,1:]

# slippting the data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# fitting Multiple linear Regression into Training data
# First we train the model on training data
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,y_train)
#Predicting the test set results
# Second we use the train model and implement it on test data to get the output of test data
y_pred= regressor.predict(X_test)
# to eradicate unnessary independent variable we use back estimation
#Building the optimal model with Backward Elimination
import statsmodels.api as sm 
X= np.append(arr = np.ones((50,1)).astype(int), values=X, axis=1)
# Creating a new matrix of features for optimal solution
X_opt = X[:,[0,1,2,3,4,5]]
X_opt = np.array(X_opt, dtype=float)
#Building a new reagressor for this specific class
regressor_ols = sm.OLS(endog = y, exog=X_opt)
regressor_ols=regressor_ols.fit()
regressor_ols.summary()
# Removing the index 2 has it has highest significance value of 0.99
X_opt = X[:,[0,1,3,4,5]]
X_opt = np.array(X_opt, dtype=float)
regressor_ols = sm.OLS(endog = y, exog=X_opt)
regressor_ols=regressor_ols.fit()
regressor_ols.summary()
# Removing the index 1  has it has highest significance value of 0.94
X_opt = X[:,[0,3,4,5]]
X_opt = np.array(X_opt, dtype=float)
regressor_ols = sm.OLS(endog = y, exog=X_opt)
regressor_ols=regressor_ols.fit()
regressor_ols.summary()
# Removing the index 4  has it has highest significance value of 0.62
X_opt = X[:,[0,3,5]]
X_opt = np.array(X_opt, dtype=float)
regressor_ols = sm.OLS(endog = y, exog=X_opt)
regressor_ols=regressor_ols.fit()
regressor_ols.summary()
# Removing the index 5  has it has highest significance value of 0.60
X_opt = X[:,[0,3]]
X_opt = np.array(X_opt, dtype=float)
regressor_ols = sm.OLS(endog = y, exog=X_opt)
regressor_ols=regressor_ols.fit()
regressor_ols.summary()
# Hence we figured out that the most R&D spend is the most important feature to consider for 
#invesment as it plays a key role in generating Profit 
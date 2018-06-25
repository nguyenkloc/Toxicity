# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
os.chdir("/Users/a-an/Desktop/Vietherb/Code/Machine learning")
#Import the data set
dataset = pd.read_csv('day.csv')
X= dataset.iloc[:,2:-1].values
y= dataset.iloc[:,15].values

# Encoding categorical data
# Encoding the Independant Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0 ] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features =[0])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X = X[:,1:]

#Splitting the dataset into the Training set and Test set  
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size =0.2, random_state=0 )

#Fitting the mutiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)  

#Predicting the Test set results
y_pred= regressor.predict(X_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

y_pred= regressor.predict(X)
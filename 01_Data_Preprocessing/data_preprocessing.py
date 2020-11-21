# data preprocessing

# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Dataset
dF = pd.read_csv('Data.csv')

# Creating Data Frames of Features and Class
X = dF.iloc[:, :-1].values
y = dF.iloc[:, 3].values

# print(X)
# print(y)

# Managing missing data in dataset
# Importing the Library required for managing missing data
from sklearn.preprocessing import Imputer

imputer1 = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer2 = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

'''
# Simply if we want to use same strategy on every column where the missing data is present we can use this :
imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
print(X)
'''

# But if we want to use different strategies on different columns we do this

# for salary column
x1 = X[:, 2].reshape(-1, 1)                             # creating a new reshaped multi-dimensional array from 1D array
x1 = imputer1.fit_transform(x1)                         # fitting and transforming data
# print(x1)
x1 = x1.reshape(1, -1)                                  # restoring to 1D array after missing data management
X = np.delete(X, 2, 1)                                  # deleting the column with missing values
X = np.insert(X, 2, x1, axis = 1)                       # inserting the column with no missing values
# print(X)

# for age column
x2 = X[:, 1].reshape(-1, 1)                             # creating a new reshaped multi-dimensional array from 1D array
x2 = imputer2.fit_transform(x2)
# print(x2)
x2 = x2.reshape(1, -1)                                  # restoring to 1D array after missing data management
X = np.delete(X, 1, 1)                                  # deleting the column with missing values
X = np.insert(X, 1, x2, axis = 1)                       # inserting the column with no missing values
# print(X)

# Encoding dataframe X
# Importing Library/Class
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# print(X)

# Importing Library/Class
from sklearn.preprocessing import OneHotEncoder
onehotencoder_X = OneHotEncoder(categorical_features = [0])
X = onehotencoder_X.fit_transform(X).toarray()
# toarray() is used to convert output to array type
print(X)

# Encoding dataframe y
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print(y)

# Splitting the Data into training set and testing set
# Importing Library
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.80)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# Feature Scaling
# Importing libraries
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler(with_mean = False)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print(X_train)
print(X_test)

# See data_preprocessing_template.py file for most important parts required to preprocess data
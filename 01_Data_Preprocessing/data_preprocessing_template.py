# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dF = pd.read_csv('Data.csv')
# Separating Features and Class
X = dF.iloc[:, :-1].values
y = dF.iloc[:, 3].values

# Splitting the dataset into training set and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''

# this template is created only to use it further in next models as preprocessing part
# See dataPreprocessing.py file for data preprocessing in detail
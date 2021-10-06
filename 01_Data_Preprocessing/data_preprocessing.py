# Importing the Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing Data Set
dF = pd.read_csv('Data.csv')
print(dF.head())

# Splitting Feature Set and Class Set
X = dF.iloc[:, :-1].values
y = dF.iloc[:, 3].values

# Managing missing data in dataset
imputer1 = Imputer(missing_values='NaN', strategy='mean', axis=0)               # Instant for Mean Methodology
imputer2 = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)      # Instant for Mode Methodology

# Salary Column

# Transforming Data
x1 = X[:, 2].reshape(-1, 1)
x1 = imputer1.fit_transform(x1)
x1 = x1.reshape(1, -1)

# Replacing Missing Value Containing Feature with Converted Feature
X = np.delete(X, 2, 1)
X = np.insert(X, 2, x1, axis = 1)

# Age Column

# Transforming Data
x2 = X[:, 1].reshape(-1, 1)
x2 = imputer2.fit_transform(x2)
x2 = x2.reshape(1, -1)

# Replacing Missing Value Containing Feature with Converted Feature
X = np.delete(X, 1, 1)
X = np.insert(X, 1, x2, axis = 1)

# Label Encoding
labelencoder = LabelEncoder()                       # Intantiating LabelEncoder
X[:, 0] = labelencoder.fit_transform(X[:, 0])       # Encoding Feature Sets
y = labelencoder.fit_transform(y)                   # Encoding Class Set

# One Hot Encoding
onehotencoder_X = OneHotEncoder(categorical_features = [0])     # Intantiating OneHotEncoder
X = onehotencoder_X.fit_transform(X).toarray()                  # Encoding Data

# Splitting data into Training Set and Testing Set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.80)

# Feature Scaling
sc_X = StandardScaler(with_mean = False)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
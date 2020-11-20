# data preprocessing

# Importing the Libraries
import numpy as np                                      # numpy contains defn. of mathematical tools and formulae
import pandas as pd                                     # pandas is used to import and manage the datasets
import matplotlib.pyplot as plt                         # matplotlib.pyplot is used to visualize the data

# Importing the Dataset
dF = pd.read_csv('Data.csv')                            # importing Data.csv file to python
# instead of using the whole dataset, we create dataframes for further processing
X = dF.iloc[:, :-1].values                              # creating an independent variable X to separate independent features
y = dF.iloc[:, 3].values                                # creating a dependent variable y to separate dependent features

#print(X)
#print(y)

# Managing missing data in dataset
# Importing the Library required for managing missing data
from sklearn.preprocessing import Imputer               # sklearn contains the documentation and definitions to make advance ML models
# preprocessing library is used to preprocess the data
# Imputer class is used to manage missing data
imputer1 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) # creating imputer1 object of Imputer class
imputer2 = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
# missing_values = 'NaN' indicates the location where data contains NaN(Not a Number or Missing Numeric Data) as values
# strategy = 'mean' indicates it is going to replace the NaN with mean of the values
# strategy = 'most_frequent' indicates it is going to replace the NaN with the value which is occurring most_frequently(mode)
# axis = 0 indicates that the values it is going to use for calculating mean is from respective "column"(axis = 1 for "row") where NaN is present as value

'''
# Simply if we want to use same strategy on every column where the missing data is present we can use this :
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)        # creating imputer object of Imputer class
# strategy = 'median' indicates it is going to replace the NaN with median of the values
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])            # replacing missing values with median of their respective columns
print(X)
'''

# But if we want to use different strategies on different columns we do this

# for salary column
x1 = X[:, 2].reshape(-1, 1)                             # creating a new reshaped multi-dimensional array from 1D array
x1 = imputer1.fit_transform(x1)                         # replacing the missing data with mean
#print(x1)
x1 = x1.reshape(1, -1)                                  # restoring to 1D array after missing data management
X = np.delete(X, 2, 1)                                  # deleting the column with missing values
X = np.insert(X, 2, x1, axis = 1)                       # inserting the column with no missing values
#print(X)

# for age column
x2 = X[:, 1].reshape(-1, 1)                             # creating a new reshaped multi-dimensional array from 1D array
x2 = imputer2.fit_transform(x2)                         # replacing the missing data with most_frequent(mode)
#print(x2)
x2 = x2.reshape(1, -1)                                  # restoring to 1D array after missing data management
X = np.delete(X, 1, 1)                                  # deleting the column with missing values
X = np.insert(X, 1, x2, axis = 1)                       # inserting the column with no missing values
#print(X)

# Since ML models can't accept categorical data because they work on mathematical formulae,
# so we need to convert the categorical data into a numeric data, for that we do this :
# Encoding X
# Importing Library/Class
from sklearn.preprocessing import LabelEncoder          # LabelEncoder class is used to manage categorical data
labelencoder_X = LabelEncoder()                         # creating object of LabelEncoder class
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])         # fitting and transforming data
#print(X)

# Since LabelEncoder replace values of categorical feature with single value thus,
# sometimes the feature it convert contains the values which are not to be graded,
# but when LabelEncoder encode the data, it replace categorical data with numeric data, i.e 0, 1, 2, ... so on,
# In mathematical term 2 is greater than 1 and 0, also 1 is greater than 0, thus ML models will think of data as a prioritized data,
# but in plain form the data was not one which was actually graded or show any priority level.
# So to solve this problem, the idea of dummy variables is used,
# and to do that we use an another class of preprocessing library
# Importing Library/Class
from sklearn.preprocessing import OneHotEncoder         # OneHotEncoder class is used to manage categorical data
onehotencoder_X = OneHotEncoder(categorical_features = [0])     # creating object of OneHotEncoder class
# categorical_features parameter take the value of the index of the column which contains categorical feature
# and is supposed to be encoded using OneHotEncoder
X = onehotencoder_X.fit_transform(X).toarray()
# toarray() function is used to convert output to array type, without this the output we are receiving is distorted
print(X)
# OneHotEncoder creates n(number of categorical features) dummy variables using the encoded data from LabelEncoder,
# all of these dummy variable have separate columns assigned to them in the dataframe.
# Wherever the value of a particular dummy variable matches the actual data the value of that column for particular row is set to 1,
# and rest of the values for dummy variable of that particular row is set to 0.

# Encoding dataframe y
# Since dataframe y contains data which can show priority level
# so encoding it's catogorical feature with LabelEncoder doesn't change the meaning
labelencoder_y = LabelEncoder()                         # creating object of LabelEncoder class
y = labelencoder_y.fit_transform(y)                     # fitting and transforming data
print(y)

# To assess the model we need to test it, thus to test it we use training set and testing set
# Splitting the Data into training set and testing set
# Importing Library
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.80)
# X_train is training variable for features
# y_train is training variable for class
# X_test is testing variable for features
# y_test is testing variable for class
# Model will train using X_train and y_train
# Model will be assessed using X_test and it's result will be checked using y_test
# train_size = 0.80 indicates that the training data set will contain 80% observation of original data set and
# rest of the size, i.e. 20% or 0.2 will be used for testing
#print(X_train)
#print(X_test)
#print(y_train)
#print(y_test)

# Since Machine-Learning-Algorithms models use Euclidean Distance thus the values of features for a respective tuple should lie in same range
# if not then the value which is much smaller will be neglected,
# Example if one column of dataset contains 50 and 24 as value and next column of dataset have 42000 and 20000 as values,
# if we apply Euclidean Distance in this dataset [(42000)^2 - (20000)^2] + [(50)^2 - (24)^2],
# thus when compared, [(50)^2 - (24)^2] seen as negligible values against [(42000)^2 - (20000)^2],
# also some models are not based on Euclidean Distance but in those cases if feature scaling is not done
# the time they will take to execute will be much much higher,
# thus to put all the values in same range we use:
# Feature Scaling
# Importing libraries
from sklearn.preprocessing import StandardScaler                    # used for Feature Scaling
sc_X = StandardScaler(with_mean = False)                            # creating object of class StandardScaler class
X_train = sc_X.fit_transform(X_train)                               # fitting and transforming our data
X_test = sc_X.transform(X_test)                                     # only transforming, because sc_X is already fitted to X_train and we have no need of fitting it to X_test again
print(X_train)
print(X_test)

# See dataPreprocessing.py file for most important parts required to preprocess data
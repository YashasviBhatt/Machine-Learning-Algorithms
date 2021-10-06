# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statsmodels.api as sm

# Importing Dataset
dF = pd.read_csv('50_Startups.csv')

# Creating Data Frames of Features and Class
X = dF.iloc[:, :-1].values
y = dF.iloc[:, 4].values

# Encoding Categorical Variables

# First Encoding using Label Encoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

# Now Encoding Encoded Variable X using One Hot Encoder
onehotencoder_X = OneHotEncoder(categorical_features = [3])
X = onehotencoder_X.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]                                        # Removing First Column

# Building Optimal Model using Backward Elimination

# adding a column with all values as ones
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

# # creating a feature variable with only high impact independent variables
# X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# regressor = sm.OLS(endog=y, exog=X_opt).fit()

# # creating a feature variable with only high impact independent variables
# X_opt = X[:, [0, 1, 3, 4, 5]]
# regressor = sm.OLS(endog=y, exog=X_opt).fit()

# # creating a feature variable with only high impact independent variables
# X_opt = X[:, [0, 3, 4, 5]]
# regressor = sm.OLS(endog=y, exog=X_opt).fit()

# # creating a feature variable with only high impact independent variables
# X_opt = X[:, [0, 3, 5]]
# regressor = sm.OLS(endog=y, exog=X_opt).fit()

# creating a feature variable with only high impact independent variables
X_opt = X[:, [0, 3]]
regressor = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor.summary())
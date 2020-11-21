# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import r2_score

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

# Splitting Dataset into Training Set and Testing Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fitting Multiple Linear Regression Model to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test Set Results
y_pred = regressor.predict(X_test)

# Evaluating Model using R-Squared Method
r2 = r2_score(y_test, y_pred)
print(r2)
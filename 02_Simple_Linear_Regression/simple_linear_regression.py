# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Importing Dataset
dF = pd.read_csv('Salary_Data.csv')

# Creating Data Frames of Features and Class
X = dF.iloc[:, :-1].values
y = dF.iloc[:, 1].values

# Splitting Dataset into Training Set and Testing Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3)

# Fitting Simple Linear Regression Model to Training Set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test Set Results
y_pred = regressor.predict(X_test)

# Visualizing Training Set Results
plt1.scatter(X_train, y_train, color = 'red', label='Scatter Plot')
plt1.plot(X_train, regressor.predict(X_train), label='Regression Line')

plt1.title('Experience vs Salary (Training Set)')
plt1.xlabel('Years of Experience')
plt1.xlabel('Salary')

plt1.show()

# Visualizing Test Set Results
plt2.scatter(X_test, y_test, color = 'red', label='Scatter Plot')
plt2.plot(X_train, regressor.predict(X_train), label='Regression Line')

plt2.title('Experience vs Salary (Test Set)')
plt2.xlabel('Years of Experience')
plt2.xlabel('Salary')

plt2.show()

# Evaluating Model using R-Squared Method
r2 = r2_score(y_test, y_pred)
print(r2)
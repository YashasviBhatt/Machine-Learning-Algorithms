# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
from sklearn.tree import DecisionTreeRegressor

# Importing Dataset
dF = pd.read_csv('Position_Salaries.csv')

# Creating Data Frames of Features and Class
X = dF.iloc[:, 1:-1].values
y = dF.iloc[:, 2:].values

# Fitting the Decision Tree Regression Model to Data
regressor = DecisionTreeRegressor()
regressor.fit(X, y)

# Predicting Result
print(regressor.predict(np.array([[6.5]])))

# Visualizing Decision Tree Regression Results
plt1.scatter(X, y, color='red', label='Scatter Plot')
plt1.plot(X, regressor.predict(X), label='Regression Line')
plt1.title('Position Level vs Salary (False)')
plt1.xlabel('Position Level')
plt1.ylabel('Salary')
plt1.show()

# Visualizing Decision Tree Regression Results (more continuous graph)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt2.scatter(X, y, color='red', label='Scatter Plot')
plt2.plot(X_grid, regressor.predict(X_grid), label='Regression Line')
plt2.title('Position Level vs Salary (Truth)')
plt2.xlabel('Position Level')
plt2.ylabel('Salary')
plt2.show()
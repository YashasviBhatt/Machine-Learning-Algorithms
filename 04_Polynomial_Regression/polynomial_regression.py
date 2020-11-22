# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing Dataset
dF = pd.read_csv('Position_Salaries.csv')

# Creating Data Frames of Features and Class
X = dF.iloc[:, 1:-1].values
y = dF.iloc[:, 2].values

# Fitting Linear Regression to Data
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to Data
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizing Linear Regression Results
plt1.scatter(X, y, color='red', label='Scatter Plot')
plt1.plot(X, lin_reg.predict(X), label='Regression Line')
plt1.title('Position Level vs Salary (Bluff)')
plt1.xlabel('Position Level')
plt1.ylabel('Salary')
plt1.show()

# Visualizing Polynomial Regression Results
plt2.scatter(X, y, color='red', label='Scatter Plot')
plt2.plot(X, lin_reg_2.predict(X_poly), label='Regression Line')
plt2.title('Position Level vs Salary (Truth)')
plt2.xlabel('Position Level')
plt2.ylabel('Salary')
plt2.show()

# Visualizing Polynomial Regression Results (more continuous graph)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt3.scatter(X, y, color='red', label='Scatter Plot')
plt3.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), label='Regression Line')
plt3.title('Position Level vs Salary (Truth)')
plt3.xlabel('Position Level')
plt3.ylabel('Salary')
plt3.show()
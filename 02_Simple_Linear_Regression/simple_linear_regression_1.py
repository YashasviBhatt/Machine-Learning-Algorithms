# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset
dF = pd.read_csv('Salary_Data.csv')

# Creating Data Frames of Features and Class
X = dF.iloc[:, 0].values
y = dF.iloc[:, 1].values

# Calculating Mean of X and y
mean_X = np.mean(X)
mean_y = np.mean(y)

# Calculating Slope of Line -- b1
numer = 0
denom = 0

for obs in range(len(X)):
    numer += ((X[obs] - mean_X) * (y[obs] - mean_y))
    denom += (X[obs] - mean_X) ** 2

b1 = numer / denom

# Calculating Intercept -- b0
b0 = mean_y - (b1 * mean_X)

# Calculating y_pred
y_pred = b0 + (b1 * X)

# Plotting Graph
plt.plot(X, y_pred, label='Regression Line')
plt.scatter(X, y, color='red', label='Scatter Plot')

plt.title('Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

plt.show()

# Evaluating Model using R-Squared Method
numer = 0
denom = 0

for obs in range(len(y)):
    numer += (y_pred[obs] - mean_y) ** 2
    denom += (y[obs] - mean_y) ** 2

r2 = numer / denom

print(r2)
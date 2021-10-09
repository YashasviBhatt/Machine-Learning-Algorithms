# Importing the Libraries
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Importing the dataset
dF = pd.read_csv('Social_Network_Ads.csv')
# Separating Features and Class
X = dF.iloc[:, 2:-1].values
y = dF.iloc[:, 4].values

# Splitting the dataset into training set and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Support Vector Machine Model to Training Set
classifier = SVC(kernel='linear')               # using linear kernel
classifier.fit(X_train, y_train)

# Predicting Test Set Results
y_pred = classifier.predict(X_test)

# Evaluating Test Set Results
cm = confusion_matrix(y_test, y_pred)
# print(((cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])))
print(accuracy_score(y_test, y_pred))
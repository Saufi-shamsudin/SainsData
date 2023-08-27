import pandas as pd
import numpy as np

zoo = pd.read_csv("C:\\Users\\user\\Desktop\\data Sc with R & Pyhton\\Zoo (1).csv")
zoo.head()
zoo.describe()
zoo.info()


zoo1 = zoo.iloc[:, 1:18] # Excluding id column

X = np.array(zoo1.iloc[:,:]) # Predictors 
Y = np.array(zoo['animal name'])  # Target 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score
pd.crosstab(Y_test, pred, rownames=['Actual'],colnames= ['Predictions']) 
print(accuracy_score(Y_test, pred))

# error on train data
pred_train = knn.predict(X_train)
pd.crosstab(Y_train, pred_train, rownames=['Actual'],colnames= ['Predictions']) 
print(accuracy_score(Y_train, pred_train))

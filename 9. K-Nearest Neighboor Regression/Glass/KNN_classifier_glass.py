import pandas as pd
import numpy as np

glass = pd.read_csv("C:\\Users\\user\\Desktop\\data Sc with R & Pyhton\\glass (1).csv")
glass.head()
glass.describe()
glass.info()


glass1 = glass.iloc[:, 0:9] # Excluding id column

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
glass2 = norm_func(glass.iloc[:,0:9])
glass2.describe()

X = np.array(glass2.iloc[:,:]) # Predictors 
Y = np.array(glass['Type'])  # Target 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
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

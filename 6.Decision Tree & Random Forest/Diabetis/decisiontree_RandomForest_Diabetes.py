import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

diabetes = pd.read_csv("C:\\Users\\user\\Desktop\\data Sc with R & Pyhton\\Diabetes (1).csv")

diabetes.head()
diabetes.info()
diabetes.describe()

#tukar nama variable (panjang sangat)
diabetes.columns = "PregQuan","PlasmaGlucose","BP","SkinTickness","Insulin2H","BMI","DBF","Age","Class"


#descrive numeric variable
diabetes.PregQuan.describe ()
diabetes.PlasmaGlucose.describe()
diabetes.BP.describe()
diabetes.SkinTickness.describe()
diabetes.Insulin2H.describe()
diabetes.BMI.describe()
diabetes.DBF.describe()
diabetes.Age.describe()
diabetes.Class.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 


#No of pregnant
plt.hist(diabetes.PregQuan) #histogram
plt.boxplot(diabetes.PregQuan) #boxplot

# plasma glucose
plt.hist(diabetes.PlasmaGlucose) #histogram
plt.boxplot(diabetes.PlasmaGlucose) #boxplot

#BP
plt.hist(diabetes.BP) #histogram
plt.boxplot(diabetes.BP) #boxplot

#skintickness
plt.hist(diabetes.SkinTickness) #histogram
plt.boxplot(diabetes.SkinTickness) #boxplot

#Insulin
plt.hist(diabetes.Insulin2H) #histogram
plt.boxplot(diabetes.Insulin2H) #boxplot

#BMI
plt.hist(diabetes.BMI) #histogram
plt.boxplot(diabetes.BMI) #boxplot

#DBF
plt.hist(diabetes.BMI) #histogram
plt.boxplot(diabetes.BMI) #boxplot

#Age
plt.hist(diabetes.Age) #histogram
plt.boxplot(diabetes.Age) #boxplot

#Class
plt.hist(diabetes.Class) #histogram
plt.boxplot(diabetes.Class) #boxplot

import seaborn as sns
sns.pairplot(diabetes.iloc[:,:])

#sebab banyak variable ada yang besar dan yangkecik kita buat normalization dulu


def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

diabetes1 = norm_func(diabetes.iloc[:,:8])
Class = diabetes['Class']

#gabung dataset baik punya

diabetes3 = pd.concat([Class, diabetes1], axis=1)


X = np.array(diabetes3.iloc[:,1:9]) # Predictors 
Y= np.array(diabetes3['Class']) # Target 



# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(X_train, Y_train)


# Prediction on Test Data
preds = model.predict(X_test)
pd.crosstab(Y_test, preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == Y_test) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(X_train)
pd.crosstab(Y_train, preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == Y_train) # Train Data Accuracy

##########RANDOM FOREST##################RANDOM FOREST##################RANDOM FOREST########
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=2, n_estimators=15, criterion="entropy")

oob_score=True,

rf.fit(X_train, Y_train)
pred = rf.predict(X_test)

from sklearn.metrics import accuracy_score

pd.crosstab(Y_test, pred, rownames=['Actual'], colnames= ['Predictions']) 
print(accuracy_score(Y_test, pred))

# test accuracy
test_acc2 = np.mean(rf.predict(X_test)==Y_test)
test_acc2

# train accuracy 
train_acc2 = np.mean(rf.predict(X_train)==Y_train)
train_acc2

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

fraud = pd.read_csv("C:\\Users\\user\\Desktop\\data Sc with R & Pyhton\\Fraud_check (1).csv")

fraud.head()
fraud.info()
fraud.describe()

#kena tukar nama kat column #ada dia letak nama variable pakai dot.. tak memasal kena tukar
fraud.columns = "undergrad","MaritalStatus","TaxableIncome","CityPopulation","WorkExp","Urban"


#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 


#taxable Income

plt.hist(fraud.TaxableIncome) #histogram
plt.boxplot(fraud.TaxableIncome) #boxplot

# citypopulation
plt.hist(fraud.CityPopulation) #histogram
plt.boxplot(fraud.CityPopulation) #boxplot

#Work experience
plt.hist(fraud.WorkExp) #histogram
plt.boxplot(fraud.WorkExp) #boxplot


import seaborn as sns
sns.pairplot(fraud.iloc[:,:])

#sebab banyak variable ada yang besar dan yangkecik kita buat normalization dulu

def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

fraud1 = norm_func(fraud.iloc[:,3:5])

#kena ubah dulu numerical data dalam Taxable income jadi categorical data
#Taxable Income
bins = [0,30000,100000]
labels = ["Risky", "Good"]
fraud['Cat_TaxIncome'] = pd.cut(fraud['TaxableIncome'], bins, labels)
fraud['TaxIncome_cat'] = pd.cut(fraud['TaxableIncome'], bins , labels = labels)

fraud2 = fraud.drop(["Cat_TaxIncome","TaxableIncome","CityPopulation","WorkExp"], axis=1)
#gabung dataset baik punya

fraud3 = pd.concat([fraud1,fraud2], axis=1)

#buat binary dulu yang mana categorical tu
lb = LabelEncoder()

fraud3["undergrad"] = lb.fit_transform(fraud3["undergrad"])
fraud3["MaritalStatus"] = lb.fit_transform(fraud3["MaritalStatus"])
fraud3["Urban"] = lb.fit_transform(fraud3["Urban"])



X = np.array(fraud3.iloc[:,:5]) # Predictors 
Y= np.array(fraud3['TaxIncome_cat']) # Target 



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

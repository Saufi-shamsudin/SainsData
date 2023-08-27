import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

company = pd.read_csv("C:\\Users\\user\\Desktop\\data Sc with R & Pyhton\\Company_Data (1).csv")

company.head()
company.info()
company.describe()

#descrive numeric variable
company.Sales.describe ()
company.CompPrice.describe()
company.Income.describe()
company.Advertising.describe()
company.Population.describe()
company.Price.describe()
company.Age.describe()
company.Education.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 


#sales
plt.hist(company.Sales) #histogram
plt.boxplot(company.Sales) #boxplot

# CompPrice
plt.hist(company.CompPrice) #histogram
plt.boxplot(company.CompPrice) #boxplot

#Income
plt.hist(company.Income) #histogram
plt.boxplot(company.Income) #boxplot

#Advertising
plt.hist(company.Advertising) #histogram
plt.boxplot(company.Advertising) #boxplot

#population
plt.hist(company.Population) #histogram
plt.boxplot(company.Population) #boxplot

#price
plt.hist(company.Price) #histogram
plt.boxplot(company.Price) #boxplot

#Age
plt.hist(company.Age) #histogram
plt.boxplot(company.Age) #boxplot

#Education
plt.hist(company.Education) #histogram
plt.boxplot(company.Education) #boxplot

import seaborn as sns
sns.pairplot(company.iloc[:,:])

#sebab kebanyakkan variable adalah numerical, kena dulu tukar kepada categorical variable

#Sales
bins = [0,8,17]
labels = ["Low", "High"]
company['Sales_cat'] = pd.cut(company['Sales'], bins, labels)
company['cat_sales'] = pd.cut(company['Sales'], bins , labels = labels)
company = company.drop(['Sales_cat'], axis = 1)

#CompPrice

bins1= [0,100,150,200]
labels1 =["below 100","101-150","more than 151"]
company['CompPrice_cat'] = pd.cut(company['CompPrice'], bins1, labels1)
company['cat_CompPrice'] = pd.cut(company['CompPrice'], bins1 , labels = labels1)

#Income
bins2= [0,40,80,120]
labels2 =["Low","Medium","High"]
company['Income_cat'] = pd.cut(company['Income'], bins2, labels2)
company['cat_Income'] = pd.cut(company['Income'], bins2 , labels = labels2)

#advertistment
bins3= [-1,10,20,30]
labels3 =["below 10","10-20","more than 20"]
company['Advertising_cat'] = pd.cut(company['Advertising'], bins3, labels3)
company['cat_advertising'] = pd.cut(company['Advertising'], bins3 , labels = labels3)

#population
bins4 = [0,200,400,510]
labels4 =["Low(below 200)","Medium(201-400)","Dense(more than 400"]
company['population_cat'] = pd.cut(company['Population'], bins4, labels4)
company['cat_population'] = pd.cut(company['Population'], bins4 , labels = labels4)

#prices
bins5 = [0,100,150,200]
labels5 =["Below 100","101-150","More than 151"]
company['price_cat'] = pd.cut(company['Price'], bins5, labels5)
company['cat_price'] = pd.cut(company['Price'], bins5 , labels = labels5)

#Age
bins6 = [0,40,60,80]
labels6 =["Below 40 years old","41-60 years old","More than 60 years old"]
company['Age_cat'] = pd.cut(company['Age'], bins6, labels)
company['cat_Age'] = pd.cut(company['Age'], bins6 , labels = labels6)

#Education
bins7 = [0,13,18]
labels7 =["Below 13","Above 14"]
company['Education_cat'] = pd.cut(company['Education'], bins7, labels)
company['cat_Education'] = pd.cut(company['Education'], bins7 , labels = labels7)

#amik variable yang nak pakai 
company1 = company[['cat_sales','cat_CompPrice','cat_Income','cat_advertising','cat_price','cat_Age','cat_Education', 'ShelveLoc','Urban','US']]



#converting into binary
lb = LabelEncoder()
company1["cat_CompPrice"] = lb.fit_transform(company1["cat_CompPrice"])
company1["cat_Income"] = lb.fit_transform(company1["cat_Income"])
company1["cat_advertising"] = lb.fit_transform(company1["cat_advertising"])
company1["cat_price"] = lb.fit_transform(company1["cat_price"])
company1["cat_Age"] = lb.fit_transform(company1["cat_Age"])
company1["cat_Education"] = lb.fit_transform(company1["cat_Education"])
company1["ShelveLoc"] = lb.fit_transform(company1["ShelveLoc"])
company1["Urban"] = lb.fit_transform(company1["Urban"])
company1["US"] = lb.fit_transform(company1 ["US"])

#categori sale kena dalam string sebab nak buat decision tree

company1['cat_sales'].unique()
company1['cat_sales'].value_counts()
colnames = list(company1.columns)
type(company1.columns)
predictors = colnames[1:]
target = colnames[0]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train,test = train_test_split(company1, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])


# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy

##########RANDOM FOREST##################RANDOM FOREST##################RANDOM FOREST########
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=2, n_estimators=15, criterion="entropy")

oob_score=True,

rf.fit(train[predictors], train[target])
pred = rf.predict(test[predictors])

from sklearn.metrics import accuracy_score

pd.crosstab(test[target], pred, rownames=['Actual'], colnames= ['Predictions']) 
print(accuracy_score(test[target], pred))

# test accuracy
test_acc2 = np.mean(rf.predict(test[predictors])==test[target])
test_acc2

# train accuracy 
train_acc2 = np.mean(rf.predict(train[predictors])==train[target])
train_acc2

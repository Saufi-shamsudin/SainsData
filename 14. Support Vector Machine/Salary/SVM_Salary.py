import pandas as pd
import numpy as np

#data testing
salary_test = pd.read_csv("C:\\Users\\user\\Desktop\\data Sc with R & Pyhton\\SalaryData_Test (3).csv")
salary_test.describe()
salary_test.info()

#data training
salary_train = pd.read_csv("C:\\Users\\user\\Desktop\\data Sc with R & Pyhton\\SalaryData_Train (3).csv")
salary_train.describe()
salary_train.info()

#describe data untuk numerical untuk dataset test

salary_test.age.describe()
salary_test.educationno.describe()
salary_test.capitalgain.describe()
salary_test.capitalloss.describe()
salary_test.hoursperweek.describe()

#describe data untuk numerical untuk dataset train

salary_train.age.describe()
salary_train.educationno.describe()
salary_train.capitalgain.describe()
salary_train.capitalloss.describe()
salary_train.hoursperweek.describe()

# EDA utk dataset test

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

#age
plt.hist(salary_test.age) #histogram
plt.boxplot(salary_test.age) #boxplot

# educationno
plt.hist(salary_test.educationno) #histogram
plt.boxplot(salary_test.educationno) #boxplot

#capitalgain
plt.hist(salary_test.capitalgain) #histogram
plt.boxplot(salary_test.capitalgain) #boxplot

#capital loss
plt.hist(salary_test.capitalloss) #histogram
plt.boxplot(salary_test.capitalloss) #boxplot

#hoursperweek
plt.hist(salary_test.hoursperweek) #histogram
plt.boxplot(salary_test.hoursperweek) #boxplot

# EDA utk dataset traning pulak


#age
plt.hist(salary_train.age) #histogram
plt.boxplot(salary_train.age) #boxplot

# educationno
plt.hist(salary_train.educationno) #histogram
plt.boxplot(salary_train.educationno) #boxplot

#capitalgain
plt.hist(salary_train.capitalgain) #histogram
plt.boxplot(salary_train.capitalgain) #boxplot

#capital loss
plt.hist(salary_train.capitalloss) #histogram
plt.boxplot(salary_train.capitalloss) #boxplot

#hoursperweek
plt.hist(salary_train.hoursperweek) #histogram
plt.boxplot(salary_train.hoursperweek) #boxplot

#pilih variable untuk di normalization dataset test

df= pd.DataFrame(salary_test)

salary_test1= df[['age','educationno','capitalgain','capitalloss','hoursperweek']]

#scaterplot dataset test

import seaborn as sns
sns.pairplot(salary_test1.iloc[:,:])

#normalization dulu test

def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

salary_test2 = norm_func(salary_test1.iloc[:,:])

#pilih variable utk normalization dataset train... letih siak otak aku main test ngan train ni,dah lah banyak method 

df1= pd.DataFrame(salary_train)

salary_train1= df1[['age','educationno','capitalgain','capitalloss','hoursperweek']]

#scaterplot dataset train

import seaborn as sns
sns.pairplot(salary_train1.iloc[:,:])

#normalization dulu train

def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

salary_train2 = norm_func(salary_train1.iloc[:,:])

#kena tukar categorical data jadi binary dataset test
#converting into binary

from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
salary_test["workclass"] = lb.fit_transform(salary_test["workclass"])
salary_test["education"] = lb.fit_transform(salary_test["education"])
salary_test["maritalstatus"] = lb.fit_transform(salary_test["maritalstatus"])
salary_test["occupation"] = lb.fit_transform(salary_test["occupation"])
salary_test["relationship"] = lb.fit_transform(salary_test["relationship"])
salary_test["race"] = lb.fit_transform(salary_test["race"])
salary_test["sex"] = lb.fit_transform(salary_test["sex"])
salary_test["native"] = lb.fit_transform(salary_test["native"])


#kena tukar categorical data jadi binary dataset train

salary_train["workclass"] = lb.fit_transform(salary_train["workclass"])
salary_train["education"] = lb.fit_transform(salary_train["education"])
salary_train["maritalstatus"] = lb.fit_transform(salary_train["maritalstatus"])
salary_train["occupation"] = lb.fit_transform(salary_train["occupation"])
salary_train["relationship"] = lb.fit_transform(salary_train["relationship"])
salary_train["race"] = lb.fit_transform(salary_train["race"])
salary_train["sex"] = lb.fit_transform(salary_train["sex"])
salary_train["native"] = lb.fit_transform(salary_train["native"])

#kita drop,slicing dan gabung data untuk dataset testing
salary_test3 = salary_test.drop(['age','educationno','capitalgain','capitalloss','hoursperweek'], axis = 1)
salary_test4 = pd.concat([salary_test2,salary_test3], axis=1)

#buat benda sama untuk data training
salary_train3 = salary_train.drop(['age','educationno','capitalgain','capitalloss','hoursperweek'], axis = 1)
salary_train4 = pd.concat([salary_train2,salary_train3], axis=1)


from sklearn.svm import SVC


train_X = salary_train4.iloc[:,:13]
train_y = salary_train4.iloc[:,13]
test_X  = salary_test4.iloc[:,:13]
test_y  = salary_test4.iloc[:,13]


# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y)

accuracy_linear = np.mean(pred_test_linear==test_y)

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y)

accuracy_RBF = np.mean(pred_test_rbf==test_y)

# kernel = polynomial
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)
np.mean(pred_test_poly==test_y)

accuracy_poly = np.mean(pred_test_poly==test_y)

#kernel = sigmoid
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(train_X,train_y)
pred_test_sig = model_poly.predict(test_X)
np.mean(pred_test_sig==test_y)

accuracy_sig = np.mean(pred_test_sig==test_y)

#buat model untuk semua kernel
d1 = {'Kernel':['Linear','RBF','Polynomial','Sigmoid'],'Accuracy':[accuracy_linear,accuracy_RBF,accuracy_poly,accuracy_sig]}
Model_SVM = pd.DataFrame(d1)  
Model_SVM







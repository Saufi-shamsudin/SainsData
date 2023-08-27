import pandas as pd
import numpy as np

forest_fires = pd.read_csv("C:\\Users\\user\\Desktop\\data Sc with R & Pyhton\\forestfires (3).csv")
forest_fires.describe()
forest_fires.info()

#describe data untuk numerical
forest_fires.FFMC.describe()
forest_fires.DMC.describe()
forest_fires.DC.describe()
forest_fires.ISI.describe()
forest_fires.temp.describe()
forest_fires.RH.describe()
forest_fires.wind.describe()
forest_fires.rain.describe()
forest_fires.area.describe()

########EDA###############EDA###############EDA#######

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

#ffmc
plt.hist(forest_fires.FFMC) #histogram
plt.boxplot(forest_fires.FFMC) #boxplot

# dmc
plt.hist(forest_fires.DMC) #histogram
plt.boxplot(forest_fires.DMC) #boxplot

#dc
plt.hist(forest_fires.DC) #histogram
plt.boxplot(forest_fires.DC) #boxplot

#isi
plt.hist(forest_fires.ISI) #histogram
plt.boxplot(forest_fires.ISI) #boxplot

#temp
plt.hist(forest_fires.temp) #histogram
plt.boxplot(forest_fires.temp) #boxplot

#RH
plt.hist(forest_fires.RH) #histogram
plt.boxplot(forest_fires.RH) #boxplot

#wind
plt.hist(forest_fires.wind) #histogram
plt.boxplot(forest_fires.wind) #boxplot

#rain
plt.hist(forest_fires.rain) #histogram
plt.boxplot(forest_fires.rain) #boxplot

#area
plt.hist(forest_fires.area) #histogram
plt.boxplot(forest_fires.area) #boxplot

import seaborn as sns
sns.pairplot(forest_fires.iloc[:,2:11])

#normalization dulu

def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

forest_fires1 = norm_func(forest_fires.iloc[:,2:11])

forest_fires2 = forest_fires.drop(["month","day","FFMC","DMC","DC","ISI","temp","RH","wind","rain","area"],axis=1)

#gabung dataset
forest_fires3 = pd.concat([forest_fires1,forest_fires2], axis=1)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train,test = train_test_split(forest_fires3,test_size = 0.20)

train_X = train.iloc[:,:28]
train_y = train.iloc[:,28]
test_X  = test.iloc[:,:28]
test_y  = test.iloc[:,28]


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







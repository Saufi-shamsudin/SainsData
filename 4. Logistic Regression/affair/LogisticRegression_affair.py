import pandas as pd
import numpy as np

#Importing Data 
affairs = pd.read_csv("C:\\Users\\user\\Desktop\\data Sc with R & Pyhton\\Affairs (1).csv")
affairs.head()
affairs.info()
affairs.describe()
#removing unwated variable
affairs1 = affairs.drop(['no'], axis=1)
affairs1.head()




#describe dulu

affairs1.naffairs.describe()
affairs1.avgmarr.describe()



#eda
import matplotlib.pyplot as plt


plt.hist(affairs1.naffairs)
plt.hist(affairs1.kids)
plt.hist(affairs1.vryunhap)
plt.hist(affairs1.unhap)
plt.hist(affairs1.avgmarr)
plt.hist(affairs1.hapavg)
plt.hist(affairs1.veryhap)
plt.hist(affairs1.antirel)
plt.hist(affairs1.notrel)
plt.hist(affairs1.slghtrel)
plt.hist(affairs1.smerel)
plt.hist(affairs1.vryrel)
plt.hist(affairs1.yrsmarr1)
plt.hist(affairs1.yrsmarr2)
plt.hist(affairs1.yrsmarr3)
plt.hist(affairs1.yrsmarr4)
plt.hist(affairs1.yrsmarr5)
plt.hist(affairs1.yrsmarr6)



#create dummy variable
affairs1['affairsyesno'] = affairs1.naffairs.map({0:"no",  1 :"yes", 2: "yes", 3: "yes", 7:"yes" ,12 :"yes" })

affairs1['affairyes']=affairs1.affairsyesno.map ({"no":0,"yes":1})





#try to modelling
import statsmodels.formula.api as sm

logit_model = sm.logit('affairyes ~  kids + vryunhap + unhap + avgmarr + hapavg + vryhap + antirel + notrel + slghtrel + smerel + vryrel + yrsmarr1 + yrsmarr2 + yrsmarr3 + yrsmarr4 + yrsmarr5+ yrsmarr6', data = affairs1).fit()
logit_model.summary()

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(affairs1, test_size = 0.3) # 30% test data

# Model building 
import statsmodels.formula.api as sm
logit_model1 = sm.logit('affairyes ~  kids + vryunhap + unhap + avgmarr + hapavg + vryhap + antirel + notrel + slghtrel + smerel + vryrel + yrsmarr1 + yrsmarr2 + yrsmarr3 + yrsmarr4 + yrsmarr5+ yrsmarr6', data = train_data).fit()

#summary
logit_model1.summary()

## Evaluation of the model
predict_test = logit_model.predict(pd.DataFrame(test_data[['kids','vryunhap', 'unhap' , 'avgmarr', 'hapavg' ,'vryhap' , 'antirel' , 'notrel' ,'slghtrel' , 'smerel', 'vryrel' , 'yrsmarr1', 'yrsmarr2', 'yrsmarr3','yrsmarr4' ,'yrsmarr5','yrsmarr6']]))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cnf_test_matrix = confusion_matrix(test_data['affairyes'], predict_test > 0.5 )
cnf_test_matrix

print(accuracy_score(test_data.affairyes, predict_test > 0.5))

## Error on train data
predict_train = logit_model.predict(pd.DataFrame(train_data[['kids','vryunhap', 'unhap' , 'avgmarr', 'hapavg' ,'vryhap' , 'antirel' , 'notrel' ,'slghtrel' , 'smerel', 'vryrel' , 'yrsmarr1', 'yrsmarr2', 'yrsmarr3','yrsmarr4' ,'yrsmarr5','yrsmarr6']]))

cnf_train_matrix = confusion_matrix(train_data['affairyes'], predict_train > 0.5 )
cnf_train_matrix

print(accuracy_score(train_data.affairyes, predict_train > 0.5))

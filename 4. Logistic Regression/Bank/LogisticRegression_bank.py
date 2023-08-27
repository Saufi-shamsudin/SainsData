import pandas as pd
import numpy as np

#Importing Data # data ni data coma nak tukar jadi coloumn n row kena letak sep ";" sebab dia jenis comma 
bank = pd.read_csv("C:\\Users\\user\\Desktop\\data Sc with R & Pyhton\\Bank (1).csv", sep= ";")
bank.head()
bank.info()
bank.describe()

#describe dulu
bank.age.describe()
bank.balance.describe()
bank.duration.describe()
bank.pdays.describe()
bank.previous.describe()


#eda
import matplotlib.pyplot as plt


plt.hist(bank.job)
plt.hist(bank.marital)
plt.hist(bank.education)
plt.hist(bank.default)
plt.hist(bank.housing)
plt.hist(bank.loan)
plt.hist(bank.contact)
plt.hist(bank.poutcome)
plt.hist(bank.y)


#removing unwated variable
bank1 = bank.drop(['day','month'], axis=1)
bank1.head()

#create dummy variable
bank1['yyes'] = bank1.y.map({'yes': 1, "no": 0})
bank1['housingyes'] = bank1.housing.map({'yes': 1, "no": 0})
bank1['loanyes'] = bank1.loan.map({'yes': 1, "no": 0})
bank1['defaultyes'] = bank1.default.map({'yes': 1, "no": 0})

#drop variable lama
bank2 = bank1.drop(['y','housing','loan','default'], axis=1)

#Try check variable
bank2.yyes.describe()
plt.hist(bank2.yyes)

#try to modelling
import statsmodels.formula.api as sm

logit_model = sm.logit('yyes ~ job + marital + education + balance + contact + duration + pdays + previous + loanyes + housingyes + defaultyes', data = bank2).fit()
logit_model.summary()

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(bank2, test_size = 0.3) # 30% test data

# Model building 
import statsmodels.formula.api as sm
logit_model1 = sm.logit('yyes ~ job + marital + education + balance + contact + duration + pdays + previous + loanyes + housingyes + defaultyes', data = train_data).fit()

#summary
logit_model1.summary()

## Evaluation of the model
predict_test = logit_model.predict(pd.DataFrame(test_data[['job','marital','education','balance','contact','duration','pdays','previous','loanyes','housingyes','defaultyes']]))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cnf_test_matrix = confusion_matrix(test_data['yyes'], predict_test > 0.5 )
cnf_test_matrix

print(accuracy_score(test_data.yyes, predict_test > 0.5))

## Error on train data
predict_train = logit_model.predict(pd.DataFrame(train_data[['job','marital','education','balance','contact','duration','pdays','previous','loanyes','housingyes','defaultyes']]))

cnf_train_matrix = confusion_matrix(train_data['yyes'], predict_train > 0.5 )
cnf_train_matrix

print(accuracy_score(train_data.yyes, predict_train > 0.5))

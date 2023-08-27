# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data # kalau dia utf-8 kena encoding dia.. encoding =latin1
corolla = pd.read_csv("C:\\Users\\user\\Desktop\\data Sc with R & Pyhton\\ToyotaCorolla (1).csv", encoding='latin1')
corolla
corolla.describe()
#summary data untuk semua variables



corolla.head()





#pilih variable yang nak digunakan
df= pd.DataFrame(corolla)

corolla_df = df[['Price', 'Age_08_04','KM','HP','cc', 'Doors','Gears','Quarterly_Tax','Weight']]Corolla_df
corolla_df



# Scatter plot between the variables along with histograms
import seaborn as sns
                
                             
# Correlation matrix 
corolla_df.corr()

#describe
corolla_df.describe()

#describe semua variable
corolla_df.Price.describe()
corolla_df.Age_08_04.describe()
corolla_df.KM.describe()
corolla_df.HP.describe()
corolla_df.cc.describe()
corolla_df.Doors.describe()
corolla_df.Gears.describe()
corolla_df.Quarterly_Tax.describe()
corolla_df.Weight.describe()


# EDA dulu
#histogram,boxplot
import matplotlib.pyplot as plt

#price

plt.bar(height =corolla_df.Price, x = np.arange(900,6000,100))
plt.hist(corolla_df.Price)
plt.boxplot(corolla_df.Price)

#Age
plt.bar(height =corolla_df.Age_08_04, x = np.arange(900,6000,100))
plt.hist(corolla_df.Age_08_04) #histogram
plt.boxplot(corolla_df.Age_08_04)

#KM
plt.bar(height = corolla_df.KM, x = np.arange(900,6000,100))
plt.hist(corolla_df.KM) #histogram
plt.boxplot(corolla_df.KM)

#HP
plt.bar(height = corolla_df.HP, x = np.arange(900,6000,100))
plt.hist(corolla_df.HP) #histogram
plt.boxplot(corolla_df.HP)

#cc
plt.bar(height = corolla_df.cc, x = np.arange(10,17,1))
plt.hist(corolla_df.cc) #histogram
plt.boxplot(corolla_df.cc)

#Doors
plt.bar(height = corolla_df.Doors, x = np.arange(10,17,1))
plt.hist(corolla_df.Doors) #histogram
plt.boxplot(corolla_df.Doors)

#Gears
plt.bar(height = corolla_df.Gears, x = np.arange(10,17,1))
plt.hist(corolla_df.Gears)
plt.boxplot(corolla_df.Gears)

#Quaterly tax
plt.bar(height = corolla_df.Quarterly_Tax, x = np.arange(10,17,1))
plt.hist(corolla_df.Quarterly_Tax)
plt.boxplot(corolla_df.Quarterly_Tax)


#weight
plt.bar(height = corolla_df.Weight, x = np.arange(30,350,50))
plt.hist(corolla_df.Weight) #histogram
plt.boxplot(corolla_df.Weight)


# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data=corolla_df).fit() # regression model

ml1.summary()
#modelling


# calculating VIF's values of independent variables
rsq_age = smf.ols('Age_08_04 ~ KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data=corolla_df).fit().rsquared  
vif_age = 1/(1-rsq_age) 
vif_age

rsq_KM = smf.ols('KM ~ Age_08_04 + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data=corolla_df).fit().rsquared  
vif_KM = 1/(1-rsq_KM)

rsq_HP = smf.ols('HP ~ Age_08_04 + KM + cc + Doors + Gears + Quarterly_Tax + Weight', data=corolla_df).fit().rsquared  
vif_HP = 1/(1-rsq_HP)

rsq_cc = smf.ols('cc ~ Age_08_04 + HP + KM + Doors + Gears + Quarterly_Tax + Weight', data=corolla_df).fit().rsquared  
vif_cc= 1/(1-rsq_cc)

rsq_Doors = smf.ols('Doors ~ Age_08_04 + HP + cc + KM + Gears + Quarterly_Tax + Weight', data=corolla_df).fit().rsquared   
vif_Doors = 1/(1-rsq_Doors)

rsq_Gears = smf.ols('Gears ~ Age_08_04 + HP + cc + Doors + KM + Quarterly_Tax + Weight', data=corolla_df).fit().rsquared   
vif_Gears = 1/(1-rsq_Gears)

rsq_tax = smf.ols('Quarterly_Tax ~ Age_08_04 + HP + cc + Doors + Gears + KM + Weight', data=corolla_df).fit().rsquared   
vif_tax = 1/(1-rsq_tax)

rsq_weight = smf.ols('Weight ~ Age_08_04 + HP + cc + Doors + KM + Quarterly_Tax + Gears', data=corolla_df).fit().rsquared   
vif_weight = 1/(1-rsq_weight)
 

# Storing vif values in a data frame
d1 = {'Variables':['Age','KM','HP','cc','Doors','Gears','Tax','Weight'],'VIF':[vif_age,vif_KM,vif_HP,vif_cc,vif_Doors,vif_Gears,vif_tax,vif_weight]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As weight is having higher VIF value, we are not going to include this prediction model
# semua variable pon okay jadi pakai je model satu. semua variable VIF <10 jadi pakai semua



### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
corolla_train, corolla_test  = train_test_split(corolla_df, test_size = 0.3) # 30% test data

# preparing the model on train data 
model_train = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight',data = corolla_train).fit()

# prediction on test data set 
test_pred = model_train.predict(corolla_test)

# test residual values 
test_resid  = test_pred - corolla_test.Price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid)) # 3.16
test_rmse


# train_data prediction
train_pred = model_train.predict(corolla_train)

# train residual values 
train_resid  = train_pred - corolla_train.Price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid)) # 4.04 
train_rmse

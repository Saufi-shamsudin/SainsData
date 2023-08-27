# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
startup = pd.read_csv("C:\\Users\\user\\Desktop\\data Sc with R & Pyhton\\50_Startups (2).csv")
startup
startup.describe()
startup.columns = "RnD","admin","marketing","state","profit"

#drop column state 
df= pd.DataFrame(startup)
df.drop(["state"], axis = 1)
startup_df = df.drop(["state"], axis = 1)
startup_df

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(startup_df.iloc[:,:])
                             
# Correlation matrix 
startup_df.corr()

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('profit ~ RnD+ admin+marketing', data=startup_df).fit() # regression model

ml1.summary()
#modelling
ml2 = smf.ols('profit ~ RnD+ admin', data=startup_df).fit()
ml2.summary()
ml3 = smf.ols('profit ~ admin + marketing', data=startup_df).fit()
ml3.summary()
ml4 = smf.ols('profit ~ RnD + marketing', data=startup_df).fit()
ml4.summary()


# calculating VIF's values of independent variables
rsq_RnD = smf.ols('RnD ~ admin + marketing', data=startup_df).fit().rsquared  
vif_RnD = 1/(1-rsq_RnD) 
vif_RnD

rsq_admin = smf.ols('admin ~ RnD+marketing', data=startup_df).fit().rsquared  
vif_admin = 1/(1-rsq_admin)

rsq_marketing = smf.ols('marketing ~ RnD + admin', data=startup_df).fit().rsquared  
vif_marketing = 1/(1-rsq_marketing)
 


# Storing vif values in a data frame
d1 = {'Variables':['RnD','admin','marketing'],'VIF':[vif_RnD,vif_admin,vif_marketing]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As weight is having higher VIF value, we are not going to include this prediction model
# semua variable pon okay jadi pakai je model satu
#model 1
ml1 = smf.ols('profit ~ RnD+ admin+marketing', data=startup_df).fit()
ml1.summary()

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
startup_train, startup_test  = train_test_split(startup_df, test_size = 0.3) # 30% test data

# preparing the model on train data 
model_train = smf.ols("profit ~ RnD+ admin+marketing ", data = startup_train).fit()

# prediction on test data set 
test_pred = model_train.predict(startup_test)

# test residual values 
test_resid  = test_pred - startup_test.profit

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid)) # 3.16
test_rmse


# train_data prediction
train_pred = model_train.predict(startup_train)

# train residual values 
train_resid  = train_pred - startup_train.profit

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid)) # 4.04 
train_rmse

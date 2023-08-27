# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
computer = pd.read_csv("C:\\Users\\user\\Desktop\\data Sc with R & Pyhton\\Computer_Data (1).csv")
computer
#summary data untuk semua variables

computer.price.describe()
computer.speed.describe()
computer.hd.describe()
computer.ram.describe()
computer.screen.describe()
computer.cd.describe()
computer.multi.describe()
computer.premium.describe()
computer.ads.describe()
computer.trend.describe()

computer.head()


computer.columns = "no","price","speed","hd","ram","screen","cd","multi","premium","ads","trend"
#histogram,boxplot
import matplotlib.pyplot as plt

#price
plt.bar(height = computer.price, x = np.arange(900,6000,100))
plt.hist(computer.price) #histogram
plt.boxplot(computer.price)

#speed
plt.bar(height = computer.speed, x = np.arange(900,6000,100))
plt.hist(computer.speed) #histogram
plt.boxplot(computer.speed)

#hd
plt.bar(height = computer.hd, x = np.arange(900,6000,100))
plt.hist(computer.hd) #histogram
plt.boxplot(computer.hd)

#ram
plt.bar(height = computer.ram, x = np.arange(900,6000,100))
plt.hist(computer.ram) #histogram
plt.boxplot(computer.ram)

#screen
plt.bar(height = computer.screen, x = np.arange(10,17,1))
plt.hist(computer.screen) #histogram
plt.boxplot(computer.screen)

#cd
plt.bar(height = computer.cd, x = np.arange(10,17,1))
plt.hist(computer.cd) #histogram

#multi
plt.bar(height = computer.multi, x = np.arange(10,17,1))
plt.hist(computer.multi)

#premium
plt.bar(height = computer.premium, x = np.arange(10,17,1))
plt.hist(computer.premium)

#ads
plt.bar(height = computer.ads, x = np.arange(30,350,50))
plt.hist(computer.ads) #histogram
plt.boxplot(computer.ads)

#trends
plt.bar(height = computer.trend, x = np.arange(1,35,1))
plt.hist(computer.trend) #histogram
plt.boxplot(computer.trend)


#drop column state 
df= pd.DataFrame(computer)
df.drop(["no"], axis = 1)
computer_df = df.drop(["no"], axis = 1)
computer_df

# Scatter plot between the variables along with histograms
import seaborn as sns
                
                             
# Correlation matrix 
computer_df1.corr()

#yes no answer tu kena buat dummy variable
computer_df['cdyes'] = computer_df.cd.map({'yes': 1, "no": 0})

computer_df['multiyes'] = computer_df.multi.map({'yes': 1, "no": 0})
computer_df['premiumyes'] = computer_df.premium.map({'yes': 1, "no": 0})


# buang cd,premium n multi yg ori
df1 = pd.DataFrame(computer_df)
df1.drop(["cd","premium","multi"], axis = 1)
computer_df1 = df1.drop(["premium","cd","multi"], axis = 1)
computer_df1






# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('price ~ trend + speed + hd + ram + screen + ads + cdyes + multiyes + premiumyes', data=computer_df1).fit() # regression model

ml1.summary()
#modelling


# calculating VIF's values of independent variables
rsq_speed = smf.ols('speed ~ trend+ hd + ram + screen + ads + cdyes + multiyes + premiumyes', data=computer_df1).fit().rsquared  
vif_speed = 1/(1-rsq_speed) 
vif_speed

rsq_hd = smf.ols('hd ~ trend+ speed + ram + screen + ads + cdyes + multiyes + premiumyes', data=computer_df1).fit().rsquared  
vif_hd = 1/(1-rsq_hd)

rsq_ram = smf.ols('ram ~ trend + hd + speed + screen + ads + cdyes + multiyes + premiumyes', data=computer_df1).fit().rsquared  
vif_ram = 1/(1-rsq_ram)

rsq_screen = smf.ols('screen~ trend + hd + speed + ram + ads + cdyes + multiyes + premiumyes', data=computer_df1).fit().rsquared  
vif_screen= 1/(1-rsq_screen)

rsq_ads = smf.ols('ads ~ trend + hd + speed + screen + ram + cdyes + multiyes + premiumyes', data=computer_df1).fit().rsquared  
vif_ads = 1/(1-rsq_ads)

rsq_cdyes = smf.ols('cdyes ~ trend + hd + speed + screen + ads + ram + multiyes + premiumyes', data=computer_df1).fit().rsquared  
vif_cdyes = 1/(1-rsq_cdyes)

rsq_multiyes = smf.ols('multiyes ~ trend + hd + speed + screen + ads + cdyes + ram + premiumyes', data=computer_df1).fit().rsquared  
vif_multiyes = 1/(1-rsq_multiyes)

rsq_premiumyes = smf.ols('premiumyes ~ trend + hd + speed + screen + ads + cdyes + multiyes + ram', data=computer_df1).fit().rsquared  
vif_premiumyes = 1/(1-rsq_premiumyes)
 
rsq_trend = smf.ols('trend  ~ premiumyes + hd + speed + screen + ads + cdyes + multiyes + ram', data=computer_df1).fit().rsquared  
vif_trend = 1/(1-rsq_trend)

# Storing vif values in a data frame
d1 = {'Variables':['speed','hd','ram','screen','ads','trend','cdyes','multiyes','premiumyes'],'VIF':[vif_speed,vif_hd,vif_ram,vif_screen,vif_ads,vif_trend,vif_cdyes,vif_multiyes,vif_premiumyes]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As weight is having higher VIF value, we are not going to include this prediction model
# semua variable pon okay jadi pakai je model satu. semua variable VIF <10 jadi pakai semua



### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
computer_train, computer_test  = train_test_split(computer_df1, test_size = 0.3) # 30% test data

# preparing the model on train data 
model_train = smf.ols('price ~ trend + speed + hd + ram + screen + ads + cdyes + multiyes + premiumyes',data = computer_train).fit()

# prediction on test data set 
test_pred = model_train.predict(computer_test)

# test residual values 
test_resid  = test_pred - computer_test.price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid)) # 3.16
test_rmse


# train_data prediction
train_pred = model_train.predict(computer_train)

# train residual values 
train_resid  = train_pred - computer_train.price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid)) # 4.04 
train_rmse

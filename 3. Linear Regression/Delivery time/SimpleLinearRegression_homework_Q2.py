# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

delivery = pd.read_csv("C:\\Users\\user\\Desktop\\data Sc with R & Pyhton\\delivery_time.csv")
delivery
delivery.columns = "del_time","sort_time"
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

delivery.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

help(plt.bar)

# del_time
plt.bar(height = delivery.del_time, x = np.arange(1,30,1))
plt.hist(delivery.del_time) #histogram
plt.boxplot(delivery.del_time) #boxplot

help(np)

# sort_time
plt.bar(height = delivery.sort_time, x = np.arange(1,10,1))
plt.hist(delivery.sort_time) #histogram
plt.boxplot(delivery.sort_time) #boxplot


# Scatter plot
plt.scatter(x=delivery['sort_time'], y=delivery['del_time'], color='green') 

# correlation
np.corrcoef(delivery.sort_time, delivery.del_time) 

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('del_time ~ sort_time', data = delivery).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(delivery['sort_time']))
pred1

# Error calculation
res1 = delivery.del_time - pred1
res_sqr1 = res1*res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1


######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at
plt.scatter(x=np.log(delivery['sort_time']),y=delivery['del_time'],color='Black')
np.corrcoef(np.log(delivery.sort_time),( delivery.del_time)) #correlation

model2 = smf.ols('del_time ~ np.log(sort_time)',data = delivery).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(delivery['sort_time']))
pred2
# Error calculation
res2 = delivery.del_time - pred2
res_sqr2 = res2*res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x=delivery['sort_time'], y=np.log(delivery['del_time']),color='red')
np.corrcoef(delivery.sort_time, np.log(delivery.del_time)) #correlation

model3 = smf.ols('np.log(del_time) ~ sort_time',data = delivery).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(delivery['sort_time']))
pred3_at = np.exp(pred3) # nak tengok data predict
pred3_at

# Error calculation
res3 = delivery.del_time - pred3_at
res3
res_sqr3 = res3*res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(del_time) ~ sort_time + I(sort_time*sort_time)', data = delivery).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(delivery))
pred4_at = np.exp(pred4)
pred4_at

# Error calculation
res4 = delivery.del_time - pred4_at
res4
res_sqr4 = res4*res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4]) }
table_rmse=pd.DataFrame(data)
table_rmse

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(delivery, test_size = 0.2)


finalmodel = smf.ols('np.log(del_time) ~ sort_time', data = train). fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_del_time = np.exp(test_pred)
pred_test_del_time

# Model Evaluation on Test data
test_res = test.del_time - pred_test_del_time
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
train_pred
pred_train_del_time = np.exp(train_pred)
pred_train_del_time

# Model Evaluation on train data
train_res = train.del_time - pred_train_del_time
train_res
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse


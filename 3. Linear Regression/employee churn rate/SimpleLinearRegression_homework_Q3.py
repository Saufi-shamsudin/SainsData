# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

emp = pd.read_csv("C:\\Users\\user\\Desktop\\data Sc with R & Pyhton\\emp_data.csv")
emp
emp.columns = "salary","churn"
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

emp.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

help(plt.bar)

# salary
plt.bar(height = emp.salary, x = np.arange(1500,2000,100))
plt.hist(emp.salary) #histogram
plt.boxplot(emp.salary) #boxplot

help(np)

# churn
plt.bar(height = emp.churn, x = np.arange(1,100,1))
plt.hist(emp.churn) #histogram
plt.boxplot(emp.churn) #boxplot


# Scatter plot
plt.scatter(x=emp['salary'], y=emp['churn'], color='purple') 

# correlation
np.corrcoef(emp.churn,emp.salary) 

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('churn ~ salary', data = emp).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(emp['salary']))
pred1

# Error calculation
res1 = emp.churn - pred1
res_sqr1 = res1*res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1


######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at
plt.scatter(x=np.log(emp['salary']),y=emp['churn'],color='Red')
np.corrcoef(np.log(emp.salary),( emp.churn)) #correlation

model2 = smf.ols('churn ~ np.log(salary)',data = emp).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(emp['salary']))
pred2
# Error calculation
res2 = emp.churn - pred2
res_sqr2 = res2*res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x=emp['salary'], y=np.log(emp['churn']),color='brown')
np.corrcoef(emp.salary, np.log(emp.churn)) #correlation

model3 = smf.ols('np.log(churn) ~ salary',data = emp).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(emp['salary']))
pred3_at = np.exp(pred3) # nak tengok data predict
pred3_at

# Error calculation
res3 = emp.churn - pred3_at
res3
res_sqr3 = res3*res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(churn) ~ salary + I(salary*salary)', data = emp).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(emp))
pred4_at = np.exp(pred4)
pred4_at

# Error calculation
res4 = emp.churn - pred4_at
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

train, test = train_test_split(emp, test_size = 0.2)


finalmodel = smf.ols('np.log(churn) ~ salary + I(salary*salary)', data = train). fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_churn = np.exp(test_pred)
pred_test_churn

# Model Evaluation on Test data
test_res = test.churn - pred_test_churn
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
train_pred
pred_train_churn = np.exp(train_pred)
pred_train_churn

# Model Evaluation on train data
train_res = train.churn - pred_train_churn
train_res
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse


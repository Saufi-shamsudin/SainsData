import pandas as pd
import datetime 
airlines= pd.read_csv("C:\\Users\\user\\Desktop\\data Sc with R & Pyhton\\Airlines Data set (2).csv")



import numpy as np
# Data Preparation
airlines["t"] = np.arange(1,97)

airlines["t_squared"] = airlines["t"]*airlines["t"]

airlines["log_Passengers"] = np.log(airlines["Passengers"])

airlines.Month.describe
# Dummy variables for Seasons

month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 


##########################
from datetime import datetime
p= airlines["Month"][0]
p[0:3]
airlines['months']= 0

for i in range(96):
   p = airlines["Month"][i]
   airlines['months'][i]= p[0:3]
   
month_dummies = pd.DataFrame(pd.get_dummies(airlines['months']))
airlines = pd.concat([airlines,month_dummies],axis = 1)

########################################
# Partition the data
airlines.iloc[:, 1].plot() # Timeplot
Train = airlines.head(77)
Test = airlines.tail(19)

####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Passengers ~ t', data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('log_Passengers ~ t', data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('Passengers ~ t + t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################

add_sea = smf.ols('Passengers ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov ', data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_Passengers ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

################## Additive Seasonality Quadratic Trend ############################

add_sea_Quad = smf.ols('Passengers ~ t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality and Exponential ###########

Mul_Add_sea = smf.ols('log_Passengers ~ t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

# so rmse_add_sea has the least value among the models prepared so far 
# Predicting new values 

model_full = smf.ols('Passengers ~ t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=airlines).fit()
prediction_passengers = pd.Series(add_sea_Quad.predict(airlines[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))


airlines1= airlines[["Month","Passengers"]]

airlines2= pd.concat([airlines1,prediction_passengers],axis = 1)
airlines2.columns= "Month","Passengers","Prediction Passengers"
airlines2.iloc[:,1:].plot() 

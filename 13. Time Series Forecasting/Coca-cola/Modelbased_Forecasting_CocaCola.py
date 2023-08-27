import pandas as pd
cocacola = pd.read_excel("C:\\Users\\user\\Desktop\\data Sc with R & Pyhton\\CocaCola_Sales_Rawdata (1).xlsx")

import numpy as np
# Data Preparation
cocacola["t"] = np.arange(1,43)

cocacola["t_squared"] = cocacola["t"]*cocacola["t"]
cocacola.columns
cocacola["log_sales"] = np.log(cocacola["Sales"])
# walmart.rename(columns={"Ridership ": 'Ridership'}, inplace=True)

# Dummy variables for Seasons
quater =['Q1','Q2','Q3','Q4'] 

p = cocacola["Quarter"][0]
p[0:2]
cocacola['qtr']= 0

for i in range(42):
   p = cocacola["Quarter"][i]
   cocacola['qtr'][i]= p[0:2]
    
month_dummies = pd.DataFrame(pd.get_dummies(cocacola['qtr']))
cocacola = pd.concat([cocacola,month_dummies],axis = 1)

# Partition the data
cocacola.iloc[:, 1].plot() # Timeplot
Train = cocacola.head(34)
Test = cocacola.tail(8)

####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales ~ t', data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('log_sales ~ t', data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('Sales ~ t + t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################

add_sea = smf.ols('Sales ~ Q1+Q2+Q3+Q4', data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1','Q2','Q3','Q4']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_sales ~ Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

################## Additive Seasonality Quadratic Trend ############################

add_sea_Quad = smf.ols('Sales ~ t+t_squared+Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1','Q2','Q3','Q4','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality and Exponential ###########

Mul_Add_sea = smf.ols('log_sales ~ t+Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

# so rmse_add_sea has the least value among the models prepared so far 
# Predicting new values 

model_full = smf.ols('Sales ~ t+t_squared+Q1+Q2+Q3+Q4', data=cocacola).fit()
pred_sales = pd.Series(add_sea_Quad.predict(cocacola[['Q1','Q2','Q3','Q4','t','t_squared']]))

cocacola1= cocacola[["Quarter","Sales"]]
cocacola2= pd.concat([cocacola1,pred_sales],axis = 1)
cocacola2.columns= "Quarter","Sales","Prediction Sales"
cocacola2.iloc[:,1:].plot() 





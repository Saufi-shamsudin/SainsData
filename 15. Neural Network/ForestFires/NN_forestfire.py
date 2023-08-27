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

#drop variable

forest_fires1 = forest_fires.drop(["month","day","area","size_category","daysun","daymon","daytue","daywed","daythu","dayfri","daysat","monthjan","monthfeb","monthmar","monthapr","monthmay","monthjun","monthjul","monthaug","monthsep","monthoct","monthnov","monthdec"],axis=1)
area = forest_fires['area']


#sambung variable
forest_fires2 = pd.concat([area,forest_fires1], axis=1)

from sklearn.model_selection import train_test_split

train,test = train_test_split(forest_fires2,test_size = 0.25)

train_X = train.iloc[:,1:]
train_y = train.iloc[:,0]
test_X  = test.iloc[:,1:]
test_y  = test.iloc[:,0]


# Importing necessary models for implementation of ANN
from keras.models import Sequential
from keras.layers import Dense #, Activation,Layer,Lambda
a=100

cont_model = Sequential()
cont_model.add(Dense(1000, input_dim=8, activation="sigmoid"))
cont_model.add(Dense(800, activation="tanh"))
cont_model.add(Dense(1, kernel_initializer="normal"))
cont_model.compile(loss="mean_squared_error", optimizer = "adam", metrics = ["mse"])

model = cont_model
model.fit(np.array(train_X), np.array(train_y), epochs=100)

# On Test dataset
pred = model.predict(np.array(test_X))
pred = pd.Series([i[0] for i in pred])

# Accuracy
np.corrcoef(pred, test_y)

layerCount = len(model.layers)
layerCount

# On Train dataset
pred_train = model.predict(np.array(train_X))
pred_train = pd.Series([i[0] for i in pred_train])

np.corrcoef(pred_train,train_y) 

hiddenWeights = model.layers[hiddenLayer].get_weights()
lastWeights = model.layers[lastLayer].get_weights()



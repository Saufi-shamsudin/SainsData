########################## Neural Network for predicting continuous values ###############################
import numpy as np
import pandas as pd

# Importing necessary models for implementation of ANN
from keras.models import Sequential
from keras.layers import Dense #, Activation,Layer,Lambda
a=100

# Reading data 
startup = pd.read_csv("C:\\Users\\user\\Desktop\\data Sc with R & Pyhton\\50_Startups (2).csv")
startup.head()

from sklearn.model_selection import train_test_split

X = startup.drop(["Profit","State"],axis=1)
Y = startup["Profit"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

cont_model = Sequential()
cont_model.add(Dense(50, input_dim=3, activation="relu"))
cont_model.add(Dense(250, activation="relu"))
cont_model.add(Dense(1, kernel_initializer="normal"))
cont_model.compile(loss="mean_squared_error", optimizer = "adam", metrics = ["mse"])

model = cont_model
model.fit(np.array(X_train), np.array(y_train), epochs=10)

# On Test dataset
pred = model.predict(np.array(X_test))
pred = pd.Series([i[0] for i in pred])

# Accuracy
np.corrcoef(pred, y_test)

layerCount = len(model.layers)
layerCount

# On Train dataset
pred_train = model.predict(np.array(X_train))
pred_train = pd.Series([i[0] for i in pred_train])

np.corrcoef(pred_train, y_train) #this is just because some model's count the input layer and others don't


#getting the weights:
hiddenWeights = model.layers[hiddenLayer].get_weights()
lastWeights = model.layers[lastLayer].get_weights()


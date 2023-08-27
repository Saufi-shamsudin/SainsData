import pandas as pd
import matplotlib.pylab as plt



airlines = pd.read_csv("C:\\Users\\user\\Desktop\\EastWestAirlines (3).csv")

airlines.describe()
airlines.info

airlines.columns =('id','bal','qualmiles','cc1miles','cc2miles','cc3miles','bonmiles','bonustrans','flightmiles12','flighttran12','daysince','award')
airlines1= airlines.drop(['id'],axis = 1)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
airlines2 = norm_func(airlines1.iloc[:, :10])
airlines2.describe()

###### scree plot or elbow curve ############
from sklearn.cluster import	KMeans

TWSS = []
k = list(range(2,8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(airlines2)
    TWSS.append(kmeans.inertia_)

TWSS
# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters=4)
model.fit(airlines2)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
airlines1['clust'] = mb # creating a  new column and assigning it to new column 

airlines1.head()

airlines1.iloc[:,0:10].groupby(airlines1.clust).mean()
airlines1.to_csv("Kmeans_airlines.csv",encoding="utf-8")

import os
os.getcwd()




import pandas as pd
import matplotlib.pylab as plt



crime = pd.read_csv("C:\\Users\\user\\Desktop\\data Sc with R & Pyhton\\crime_data (2).csv")

crime.describe()
crime.info

#EDA
#age
plt.hist(crime.Murder) #histogram
plt.boxplot(crime.Murder) #boxplot

# educationno
plt.hist(crime.Assault) #histogram
plt.boxplot(crime.Assault) #boxplot

#capitalgain
plt.hist(crime.Urbanpop) #histogram
plt.boxplot(crime.Urbanpop) #boxplot

#capital loss
plt.hist(crime.Rape) #histogram
plt.boxplot(crime.Rape) #boxplot

#letak nama dulu pasal coloumn kosong takde nama

crime.columns = "Location","Murder","Assault","Urbanpop","Rape"


# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
crime1= norm_func(crime.iloc[:, 1:])

###### scree plot or elbow curve ############
from sklearn.cluster import	KMeans

TWSS = []
k = list(range(1,5))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(crime1)
    TWSS.append(kmeans.inertia_)

TWSS
# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters=2)
model.fit(crime1)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
crime['clust'] = mb # creating a  new column and assigning it to new column 

crime.head()

crime.iloc[:,1:5].groupby(crime.clust).mean()
crime.to_csv("Kmeans_crime.csv",encoding="utf-8")

import os
os.getcwd()



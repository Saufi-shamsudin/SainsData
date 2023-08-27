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



# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(crime1, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters=4, linkage = 'complete', affinity = "euclidean").fit(crime1) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

crime['cluster']=cluster_labels # creating a new column and assigning it to new column 



# Aggregate mean of each cluster
crime.iloc[:, :].groupby(crime.cluster).mean()

# creating a csv file 
crime.to_csv("University.csv", encoding = "utf-8")

import os
os.getcwd()

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

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(airlines2, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters=4, linkage = 'complete', affinity = "euclidean").fit(airlines2) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

airlines1['cluster']=cluster_labels # creating a new column and assigning it to new column 



# Aggregate mean of each cluster
airlines1.iloc[:, :].groupby(airlines1.cluster).mean()

# creating a csv file 
airlines.to_csv("University.csv", encoding = "utf-8")

import os
os.getcwd()

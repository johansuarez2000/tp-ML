#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib . pyplot as plt
import time
from sklearn import cluster
from scipy.io import arff
from sklearn import metrics
from sklearn.metrics import pairwise_distances, silhouette_samples
import matplotlib.cm as cm
import scipy.cluster.hierarchy as shc

path = './artificial/'
## Load differents datasets

## ---------------Datasets with evident clusters------------------------

#databrut = arff.loadarff(open(path+"2d-4c.arff" , 'r') )
#databrut = arff.loadarff (open(path+"xclara.arff" , 'r') )

## ______________________________________________________________________


## --------------Dataset without evident classes-----------------

#databrut = arff.loadarff (open(path+"spiral.arff" , 'r') )
databrut = arff.loadarff (open(path+"3-spiral.arff" , 'r') )

datanp = [ [ x[0] ,x[1]] for x in databrut [ 0 ] ]

f0 = [x[0] for x in datanp]
f1 = [x[1] for x in datanp]


# Donnees dans datanp
print ( " Dendrogramme ’ single ’ donnees initiales " )
linked_mat = shc.linkage(datanp , 'single')
plt.figure ( figsize = ( 12 , 12 ) )
shc.dendrogram ( linked_mat ,
orientation = 'top' ,
distance_sort = 'descending' ,
show_leaf_counts = False )
plt . show ()

# set distance_threshold ( 0 ensures we compute the full tree )
tps1 = time . time ()
model = cluster . AgglomerativeClustering ( distance_threshold = 10 , linkage = 'single' , n_clusters = None )
model = model . fit ( datanp )
tps2 = time . time ()
labels = model . labels_
k = model . n_clusters_
leaves = model . n_leaves_
# Affichage clustering
plt . scatter ( f0 , f1 , c = labels , s = 8 )
plt . title ( " Resultat du clustering " )
plt . show ()
print ( " nb clusters = " ,k , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
if (k > 1) :
    silhouette_avg=metrics.silhouette_score(datanp, labels, metric='euclidean')
    print("silhoutte: ", silhouette_avg )

#set the number of clusters
k = 3
tps1 = time . time ()
model = cluster . AgglomerativeClustering ( linkage = 'single' , n_clusters = k )
model = model . fit ( datanp )
tps2 = time . time ()
labels = model . labels_
kres = model . n_clusters_
leaves = model . n_leaves_
plt . scatter ( f0 , f1 , c = labels , s = 8 )
plt . title ( " Resultat du clustering " )
plt . show ()
print ( " nb clusters = " ,k , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
silhouette_avg=metrics.silhouette_score(datanp, labels, metric='euclidean')
print("silhoutte: ", silhouette_avg )

########### --------------------------------------------------------##################



print ( " Dendrogramme ’ average ’ donnees initiales " )
linked_mat = shc.linkage(datanp , 'average')
plt.figure ( figsize = ( 12 , 12 ) )
shc.dendrogram ( linked_mat ,
orientation = 'top' ,
distance_sort = 'descending' ,
show_leaf_counts = False )
plt . show ()

# set distance_threshold ( 0 ensures we compute the full tree )
tps1 = time . time ()
model = cluster . AgglomerativeClustering ( distance_threshold = 10 , linkage = 'average' , n_clusters = None )
model = model . fit ( datanp )
tps2 = time . time ()
labels = model . labels_
k = model . n_clusters_
leaves = model . n_leaves_
# Affichage clustering
plt . scatter ( f0 , f1 , c = labels , s = 8 )
plt . title ( " Resultat du clustering " )
plt . show ()
print ( " nb clusters = " ,k , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
if (k > 1) :
    silhouette_avg=metrics.silhouette_score(datanp, labels, metric='euclidean')
    print("silhoutte: ", silhouette_avg )
# set the number of clusters
k = 3
tps1 = time . time ()
model = cluster . AgglomerativeClustering ( linkage = 'average' , n_clusters = k )
model = model . fit ( datanp )
tps2 = time . time ()
labels = model . labels_
kres = model . n_clusters_
leaves = model . n_leaves_
plt . scatter ( f0 , f1 , c = labels , s = 8 )
plt . title ( " Resultat du clustering " )
plt . show ()
print ( " nb clusters = " ,k , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
silhouette_avg=metrics.silhouette_score(datanp, labels, metric='euclidean')
print("silhoutte: ", silhouette_avg )

########### --------------------------------------------------------##################


print ( " Dendrogramme ’ complete ’ donnees initiales " )
linked_mat = shc.linkage(datanp , 'complete')
plt.figure ( figsize = ( 12 , 12 ) )
shc.dendrogram ( linked_mat ,
orientation = 'top' ,
distance_sort = 'descending' ,
show_leaf_counts = False )
plt . show ()

# set distance_threshold ( 0 ensures we compute the full tree )
tps1 = time . time ()
model = cluster . AgglomerativeClustering ( distance_threshold = 10 , linkage = 'complete' , n_clusters = None )
model = model . fit ( datanp )
tps2 = time . time ()
labels = model . labels_
k = model . n_clusters_
leaves = model . n_leaves_
# Affichage clustering
plt . scatter ( f0 , f1 , c = labels , s = 8 )
plt . title ( " Resultat du clustering " )
plt . show ()
print ( " nb clusters = " ,k , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
if (k > 1) :
    silhouette_avg=metrics.silhouette_score(datanp, labels, metric='euclidean')
    print("silhoutte: ", silhouette_avg )

# set the number of clusters
k = 3
tps1 = time . time ()
model = cluster . AgglomerativeClustering ( linkage = 'complete' , n_clusters = k )
model = model . fit ( datanp )
tps2 = time . time ()
labels = model . labels_
kres = model . n_clusters_
leaves = model . n_leaves_
plt . scatter ( f0 , f1 , c = labels , s = 8 )
plt . title ( " Resultat du clustering " )
plt . show ()
print ( " nb clusters = " ,k , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
silhouette_avg=metrics.silhouette_score(datanp, labels, metric='euclidean')
print("silhoutte: ", silhouette_avg )


########### --------------------------------------------------------##################

print ( " Dendrogramme ’ ward ’ donnees initiales " )
linked_mat = shc.linkage(datanp , 'ward')
plt.figure ( figsize = ( 12 , 12 ) )
shc.dendrogram ( linked_mat ,
orientation = 'top' ,
distance_sort = 'descending' ,
show_leaf_counts = False )
plt . show ()

# set distance_threshold ( 0 ensures we compute the full tree )
tps1 = time . time ()
model = cluster . AgglomerativeClustering ( distance_threshold = 10 , linkage = 'ward' , n_clusters = None )
model = model . fit ( datanp )
tps2 = time . time ()
labels = model . labels_
k = model . n_clusters_
leaves = model . n_leaves_
# Affichage clustering
plt . scatter ( f0 , f1 , c = labels , s = 8 )
plt . title ( " Resultat du clustering " )
plt . show ()
print ( " nb clusters = " ,k , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
if (k > 1) :
    silhouette_avg=metrics.silhouette_score(datanp, labels, metric='euclidean')
    print("silhoutte: ", silhouette_avg )
# set the number of clusters

k = 3
tps1 = time . time ()
model = cluster . AgglomerativeClustering ( linkage = 'ward' , n_clusters = k )
model = model . fit ( datanp )
tps2 = time . time ()
labels = model . labels_
kres = model . n_clusters_
leaves = model . n_leaves_
plt . scatter ( f0 , f1 , c = labels , s = 8 )
plt . title ( " Resultat du clustering " )
plt . show ()
print ( " nb clusters = " ,k , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )   
silhouette_avg=metrics.silhouette_score(datanp, labels, metric='euclidean')
print("silhoutte: ", silhouette_avg )




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

path = './dataset-rapport/'
columna1=[]
columna2=[]

## Écrivez ici le nom du fichier .txt
#with open(path+'x1.txt', 'r') as file:
#with open(path+'x2.txt', 'r') as file:
#with open(path+'x3.txt', 'r') as file:
#with open(path+'x4.txt', 'r') as file:
#with open(path+'y1.txt', 'r') as file:
#with open(path+'zz1.txt', 'r') as file:
with open(path+'zz2.txt', 'r') as file:
    for line in file:
        columns = line.split()
        if len(columns) == 2:
            columna1.append(float(columns[0]))
            columna2.append(float(columns[1]))
            
datanp = list(zip(columna1, columna2))
f0=columna1
f1=columna2

# Donnees dans datanp
print ( " Single : " )
linked_mat = shc.linkage(datanp , 'single')
plt.figure ( figsize = ( 12 , 12 ) )
shc.dendrogram ( linked_mat ,
orientation = 'top' ,
distance_sort = 'descending' ,
show_leaf_counts = False )
plt . show ()

# set distance_threshold ( 0 ensures we compute the full tree )
tps1 = time . time ()
model = cluster . AgglomerativeClustering ( linkage = 'single' , n_clusters = 2 )
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


########### --------------------------------------------------------##################



print ( " Average :" )
linked_mat = shc.linkage(datanp , 'average')
plt.figure ( figsize = ( 12 , 12 ) )
shc.dendrogram ( linked_mat ,
orientation = 'top' ,
distance_sort = 'descending' ,
show_leaf_counts = False )
plt . show ()

# set distance_threshold ( 0 ensures we compute the full tree )
tps1 = time . time ()
model = cluster . AgglomerativeClustering (linkage = 'average' , n_clusters = 2 )
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


########### --------------------------------------------------------##################


print ( "Complete: " )
linked_mat = shc.linkage(datanp , 'complete')
plt.figure ( figsize = ( 12 , 12 ) )
shc.dendrogram ( linked_mat ,
orientation = 'top' ,
distance_sort = 'descending' ,
show_leaf_counts = False )
plt . show ()

# set distance_threshold ( 0 ensures we compute the full tree )
tps1 = time . time ()
model = cluster . AgglomerativeClustering ( linkage = 'complete' , n_clusters = 2 )
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


########### --------------------------------------------------------##################

print ( "  Ward:  " )
linked_mat = shc.linkage(datanp , 'ward')
plt.figure ( figsize = ( 12 , 12 ) )
shc.dendrogram ( linked_mat ,
orientation = 'top' ,
distance_sort = 'descending' ,
show_leaf_counts = False )
plt . show ()

# set distance_threshold ( 0 ensures we compute the full tree )
tps1 = time . time ()
model = cluster . AgglomerativeClustering (linkage = 'ward' , n_clusters = 2 )
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



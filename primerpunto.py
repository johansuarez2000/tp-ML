import numpy as np
import matplotlib . pyplot as plt
from scipy.io import arff
# Parser un fichier de donnees au format arff
# data est un tableau d ’ exemples avec pour chacun
# la liste des valeurs des features
#
# Dans les jeux de donnees consideres :
# il y a 2 features ( dimension 2 )
# Ex : [[ - 0 . 499261 , -0 . 0612356 ] ,
# [ - 1 . 51369 , 0 . 265446 ] ,
# [ - 1 . 60321 , 0 . 362039 ] , .....
# ]
#
# Note : chaque exemple du jeu de donnees contient aussi un
# numero de cluster . On retire cette information
path = './artificial/'
 
## Load differents datasets

#databrut = arff.loadarff (open(path+"xclara.arff" , 'r') )
#databrut = arff.loadarff (open(path+"cure-t2-4k.arff" , 'r') )
#databrut = arff.loadarff (open(path+"ds4c2sc8.arff" , 'r') )
#databrut = arff.loadarff (open(path+"2d-4c.arff" , 'r') )

#databrut = arff.loadarff (open(path+"spiral.arff" , 'r') )
#databrut = arff.loadarff (open(path+"impossible.arff" , 'r') )
#databrut = arff.loadarff (open(path+"banana.arff" , 'r') )
#databrut = arff.loadarff (open(path+"hypercube.arff" , 'r') )
databrut = arff.loadarff (open(path+"3-spiral.arff" , 'r') )

datanp = [ [ x[0] ,x[1]] for x in databrut [ 0 ] ]
# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0 = [ - 0 . 499261 , -1 . 51369 , -1 . 60321 , ...]
# Ex pour f1 = [ - 0 . 0612356 , 0 . 265446 , 0 . 362039 , ...]
#f0 = datanp [:,0] # tous les elements de la premiere colonne
f0 = [x[0] for x in datanp]
#f1 = datanp [:,1] # tous les elements de la deuxieme colonne
f1 = [x[1] for x in datanp]
plt.scatter (f0,f1,s = 8 )
plt.title ( "Donnees initiales" )
plt.show ()
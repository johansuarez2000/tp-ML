# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 16:18:15 2024
"""
import numpy as np
import matplotlib . pyplot as plt
from scipy.io import arff
path = './dataset-rapport/'
columna1=[]
columna2=[]
with open(path+'y1.txt', 'r') as file:
    # Lee cada línea del archivo
    for line in file:
        # Divide la línea en columnas usando el espacio como separador
        columns = line.split()
        if len(columns) == 2:
            columna1.append(float(columns[0]))
            columna2.append(float(columns[1]))
            
datanp = list(zip(columna1, columna2))
f0 = [x[0] for x in datanp]
f1 = [x[1] for x in datanp]
plt.scatter(f0,f1, s=8)
plt.title ("Donnees initiales")
plt.show ()                                                                                                                                                                                            

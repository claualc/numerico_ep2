# EP 2 DE NUMÉRICO
##  Autovalores e Autovetores de Matrizes
### Tridiagonais Simétricas

#- Claudia Baz Alvarez 11261573
#- Gustavo de Mattos Ivo Junqueira 11259402

import numpy as np
from helpers.helpers_householder import *
from helpers.trelicas import *
import sys, re

np.set_printoptions(suppress=False, precision=8, threshold=sys.maxsize)
np.set_printoptions(edgeitems=300, linewidth=10000000)

#leitura do input 
contents = open('./resources/input-c.txt', 'r').read()
c = np.array(re.split(' |\n',contents)).astype(np.float64)

#init  variables
info_barras = []
[num_nos, num_fixos, num_barras ] = c[:3]
[den, area, elast] = c[3:6]
for i in range(int(num_barras)):
    info_barras.append(
        np.array(c[i*4+6:6+4*(i+1)]))


# Calculando matrizes das treliças
K = get_K(info_barras, area, elast)
M = get_M(info_barras, den, area)

# Calculando os autovalores e autovetores de K~
M = M**(-1/2)*np.identity(24)
K_ = np.array(M@K@M)
tria, Ht = get_tridiagonal(K_)
print("HT")
print(Ht)
eigenvalues, eigenvectors, ite = qr(tria, True, 1e-6, Ht)

#frequencias
eigenvalues_5 = np.sort(eigenvalues)[0:5]
freq = eigenvalues_5**(1/2)
#modos de vibraçãotip
mod = M@eigenvectors

# RESULTADOS
print(  "\nMatriz tridiagonal - householder")
print(  np.array(tria))
print(  '\nAutovalores calculados com o algoritmo QR')
print(  eigenvalues)

print(  "\n\nMENORES FREQUÊNCIAS DE VIBRAÇÃO")
for i in range(0,5):
    index = np.where(eigenvalues == eigenvalues_5[i])
    print(  "Frequência", i+1,": ", freq[i])
    print(  "Modo de Vibração",i+1,": \n",mod.T[index], "\n")

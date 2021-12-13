# EP 2 DE NUMÉRICO
##  Autovalores e Autovetores de Matrizes
### Tridiagonais Simétricas

#- Claudia Baz Alvarez 11261573
#- Gustavo de Mattos Ivo Junqueira 11259402

import numpy as np
from helpers.helpers_householder import *
import sys
import numpy.linalg as nplg

np.set_printoptions(suppress=True, precision=5, threshold=sys.maxsize)
np.set_printoptions(edgeitems=30, linewidth=100000)


#iniciar variaveis
A = get_b_matrix(20)
num = 1

#Calculando autovalores e autovetores
tria, Ht = get_tridiagonal(np.array(A))
autovalores_analiticos = get_analitic_eigenvalues(A)
eigenvalues, eigenvectors, ite = qr(tria, True, 1e-6, Ht)

# RESULTADOS
print("\n\nITEM B - Matriz 20x20")
print("\nMatriz tridiagonal - householder")
print(tria)

print('\n\nAutovalores analiticos')
print(np.array(autovalores_analiticos))
print('\nAutovalores calculados com o algoritmo QR')
print(np.flip(np.array(eigenvalues)))

# Calculando testes e errores
eigen_error = np.zeros_like(eigenvalues)
for i,val in enumerate(eigenvalues):
    eigenvec = eigenvectors[:,i].T
    Av = A@eigenvec
    lambdav = val * eigenvec
    eigen_error[i]  = nplg.norm(Av - lambdav, ord=np.inf)

error = np.identity(eigenvectors.shape[0]) - eigenvectors.T@eigenvectors
orthonormal_error = nplg.norm(error, ord=np.inf)

print("\nMatriz de Autovetores")
print(eigenvectors.T)
print('\n\nAutovalores E Autovetores calculados')
for i, eigenval in enumerate(eigenvalues):
    print("Autovalor",num,":", eigenval)
    print("Autovetor",num,":", eigenvectors.T[:,i], "\n")
    num+= 1
print(f'\nErro máximo  entre A.v e lambda.v =', np.max(eigen_error))
print(f'\n I - V_T.V = ', orthonormal_error)





    


# EP 2 DE NUMÉRICO
##  Autovalores e Autovetores de Matrizes
### Tridiagonais Simétricas

#- Claudia Baz Alvarez 11261573
#- Gustavo de Mattos Ivo Junqueira 11259402

import numpy as np
import numpy.linalg as nplg
from helpers.helpers_householder import *
import sys

np.set_printoptions(suppress=True, precision=3, threshold=sys.maxsize)
np.set_printoptions(edgeitems=30, linewidth=100000)

#Iniciando variaveis
i = 1
A = get_a_matrix()
(n,n) = np.shape(A)

# Calculando autovalores e autovetores da matriz A
tria, Ht = get_tridiagonal(np.array(A))
eigenvalues, eigenvectors, ite = qr(tria, True, 1e-6, Ht)

# Calculando testes e errores
eigen_error = np.zeros_like(eigenvalues)
for i,val in enumerate(eigenvalues):
    eigenvec = eigenvectors[:,i].T
    Av = A@eigenvec
    lambdav = val * eigenvec
    eigen_error[i]  = nplg.norm(Av - lambdav, ord=np.inf)

error = np.identity(eigenvectors.shape[0]) - eigenvectors.T@eigenvectors
orthonormal_error = nplg.norm(error, ord=np.inf)

# RESULTADOS
print("\n\nITEM A - Matriz 4x4")
print("\nMatriz tridiagonal - householder")
print(tria)
print('\n\nAutovalores E Autovetores calculados')
for val,vec in zip(eigenvalues,eigenvectors.T):
    print("Autovalor",i ,":", val)
    print("Autovetor",i ,":", vec, "\n")
    i+= 1
print(f'\nErro máximo  entre A.v e lambda.v =', np.max(eigen_error))
print(f'\n I - V_T.V = ', orthonormal_error)


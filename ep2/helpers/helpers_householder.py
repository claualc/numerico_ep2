# EP DE NUMÉRICO
##  Autovalores e Autovetores de Matrizes
### Tridiagonais Simétricas - O Algoritmo QR

#- Gustavo de Mattos Ivo Junqueira 11259402
#- Claudia Baz Alvarez 11261573


import numpy as np
from math import sqrt
from helpers.helpers_qr import *
import sys

def get_analitic_eigenvalues(A):
    eigenvalues = []
    (n,n) = np.shape(A)
    for i in range(1,n+1):
        val = 1/2*1/(1 - np.cos((2*i -1)*np.pi/(2*n+1)) )
        eigenvalues.append(val)
    return eigenvalues

def get_a_matrix():
    return np.array([
        [2,4,1,1],
        [4,2,1,1],
        [1,1,1,2],
        [1,1,2,1],
    ])

# def get_a_matrix():
#     return np.array([
#         [2,-1,1,3],
#         [-1,1,4,2],
#         [1,4,2,-1],
#         [3,2,-1,1],
#     ])

def get_b_matrix(n):
    A = np.zeros(shape=(n,n))
    for i in range(n):
        A[i, :] = n - i
        A[:, i] = n - i
    return A

#i - coluna
def obter_w(x, delta):
    e2 = np.zeros_like(x)
    e2[0] = 1 
    w = x + e2*np.linalg.norm(x)*np.sign(delta)
    return w

def get_tridiagonal(A):

    (n,n) = np.shape(A)
    HwXHw = np.empty(shape=(n,n))
    HwX = np.empty(shape=(n,n))
    Ht = np.identity(n)
    # igualar X À matriz A
    X = np.empty(shape=(n,n))
    for j in range(n):
        X[j, :] = A[j, :] 

    #retorna a transformação de householder de um vetor X
    def householder(x, x_0, w):
        v = [ x_0 ]
        v.extend(x - 2*np.dot(w,x)/np.dot(w,w)*w)
        return np.array(v)

    # For each iteration, H matrix is calculated for (i+1)th row
    for i in range(n-2):
        if i < n-1 :

            #obter a transformação do vetor a1 definido no enunciado
            w = obter_w(
                X[i+1:, i], #a1
                X[i+1, i]   #delta
            )

            # HwX
            for j in range(len(w)+1):
                HwX[j+i, i:] = householder(
                    X[i+1:, j+i], #vetor x: todos os items da coluna j da matriz X menos o 1+o numero da iteração
                    X[i, j+i],    #v_0: guarda a primeira linha da matriz X
                    w )

            # tendo HwX e X --> "Hw = HwX / X" 
            invX = np.linalg.inv(X)  
            Ht[:,i:]= (Ht@invX@HwX)[:,i:]

            # HwXHw 
            for j in range(len(w)+1):
                HwXHw[j+i, i:] = householder(
                    HwX[i+1:, j+i], #vetor x: todos os items da coluna j da matriz X menos o 1+o numero da iteração
                    HwX[i, j+i],     #v_0: guarda a primeira linha da matriz X
                    w )

        X = HwXHw  
    return X, Ht


def get_k_matrix(barra, area, elast):
    [a, b, angulo, comp] = barra
    c = np.cos(np.deg2rad(angulo))
    s = np.sin(np.deg2rad(angulo))
    
    matriz_values = area*elast*1e9/comp*np.array([
    [c*c, c*s,-c*c, -c*s ],
    [c*s, s*s,-c*s, -s*s],
    [-c*c, -c*s, c*c, c*s],
    [-c*s, -s*s, c*s, s*s]
    ])
    matrix_indexs = np.array([
         [    (2*a-1,2*a-1),     (2*a-1,2*a),      (2*a-1,2*b-1),    (2*a-1,2*b)],
         [      (2*a,2*a-1),       (2*a,2*a),        (2*a,2*b-1),      (2*a,2*b)],
         [    (2*b-1,2*a-1),     (2*b-1,2*a),      (2*b-1,2*b-1),    (2*b-1,2*b)],
         [      (2*b,2*a-1),       (2*b,2*a),        (2*b,2*b-1),      (2*b,2*b)]
    ])

    #index -1
    for index,item in np.ndenumerate(matrix_indexs):
        matrix_indexs[index] -= 1
    return matriz_values, matrix_indexs


   
    




    


import numpy as np
from helpers.helpers_householder import *

def get_K(info_barras, area, elast):
    K = np.zeros((24,24), dtype=float)
    for barra in info_barras:
        [a, b, angulo, comp] = barra
        Kij, indexs = get_k_matrix(barra, area, elast)

        for (values, indexs) in zip(Kij, indexs):
            #itera cada linha de Kji
            for i in range(len(values)):
                (um, dois) = indexs[i].astype(int)
                if um< 24 and dois< 24: 
                    K[ um, dois ] += values[i]
    
    return np.array(K)
    
def get_M(info_barras, den, area):
    M = np.zeros((24), dtype=float)
    nodes_matrix = np.delete(info_barras,(2,3), axis=1)
    for i in range(1,13):
        rows, cols= np.where(nodes_matrix == i)
        for row in rows:
            barra=info_barras[row]
            M[2*i-1-1] += barra[3]*den*area*0.5
            M[2*i-1] += barra[3]*den*area*0.5  
    
    return np.array(M)


        

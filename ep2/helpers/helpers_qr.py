import numpy as np

def givens_rotation(A):

    def get_givens_matrix(a, b):
    #returns Given Matrix for each iteration
        r = np.sqrt(a**2+b**2)
        c = a/r
        s = -b/r
        G = np.identity(n)
        G[[j, i], [j, i]] = c
        G[i, j] = s
        G[j, i] = -s
        return G

    [n, n] = np.shape(A)
    Q = np.identity(n)
    R = np.array(A)

    (rows, cols) = np.tril_indices(n, -1, n)
    for (i, j) in zip(rows, cols):
            G = get_givens_matrix(R[j, j], R[i, j])
            R = np.matmul(G, R)
            Q = np.matmul(Q, G.T)

    return (Q, R)


def wilkinson_heuristic(A, n):
    #calculates the shift aplied to the QR method
    def sgn(d):
        val = 1
        if d<0: val=-1
        return val

    def module(a,b):
        return np.sqrt(a**2+b**2)

    # if A shapes nxn, having n=3:
    # A = [ [ x       x         x    ] 
    #       [ x    alfa_n_1     x    ] 
    #       [ x     b_n_1     alfa_n ]  ]
    index = n-1
    alfa_n   = A[index,index]
    alfa_n_1 = A[index-1,index-1]
    b_n_1    = A[index,index-1]
    d = (alfa_n_1 - alfa_n)/2

    return alfa_n + d - sgn(d)*module(d,b_n_1) 

def printmatrix(A):
    for (i,j), cell in np.ndenumerate(A):
        A[i,j]= round(A[i,j], 2)
    np.set_printoptions(suppress=True)
    print(A)

def qr(A, add_shift, error, Q_init):
    #init vars
    AK = A
    (n,n) = np.shape(AK)
    diag = np.diag_indices(n)
    eigenvalues = []
    eigenvectors = Q_init
    ite = 0

    #when a eigenvalue if found
    #adds it to the array and deletes a dimension from AK matrix
    def get_eigen_value(AK, n):
        eigenvalues.append(AK[n-1,n-1])
        AK = np.delete(AK, n-1, 0)
        AK = np.delete(AK, n-1, 1)
        return AK

    def get_shift(AK, n):
        shift = 0
        if add_shift and ite > 0:
            shift = wilkinson_heuristic(AK, n)
        s_matrix = shift*np.identity(n)
        return s_matrix

    m = n
    while len(eigenvalues) != np.shape(A)[0]:
        (n,n) = np.shape(AK)
        while n != 1 and np.abs(AK[n-1, n-2]) > error:
            
            shift = get_shift(AK, n)
            Q,R = givens_rotation(AK - shift)
            AK = R@Q + shift
            
            eigenvectors[:,0:m] = eigenvectors[:,0:m]@Q
            ite += 1
        
        AK= get_eigen_value(AK, n)
        m-=1
    
    return np.array(eigenvalues),np.flip(eigenvectors, axis=1), ite



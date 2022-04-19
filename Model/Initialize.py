import numpy as np
from typing import NewType
import sklearn.datasets
import numpy.linalg as linalg

# functions used in ModelClass for initialization
# passed quality test, no bugs found

def init_Z(sample_Size, cluster_Count):
    Z = np.zeros((sample_Size, cluster_Count), dtype=np.double)
    membership = np.random.choice(cluster_Count, sample_Size) # randomly assign data to components
    for i in range(sample_Size):
        g = membership[i]
        Z[i, g] = 1
    return Z    # if data 10 is in component 3, then Z[10, 3] = 1

def init_MixProp(cluster_Count):
    A1 = np.ones(cluster_Count) 
    propVect = A1 / cluster_Count
    entrySum = np.dot(propVect, A1)
    if entrySum != 1:   # catch round off error accumulations!
        print('\n👇👇👇 oh no! (screams)\n', propVect, '\n👆👆👆 doesn\'t work\n')
        propVect[-1] = 1 - entrySum + propVect[-1]
        print(entrySum)
    return propVect

def init_Mean(latent_Count, cluster_Count):
    # mean as column vectors
    A =  np.random.rand(latent_Count, cluster_Count) 
    return A

def init_Psi(rawDataDimension):
    return np.diag(np.ones(rawDataDimension))

dim = NewType('dim', int)

def init_Cholesky(g, dim):
    Lg = 0 * np.random.rand(g, dim, dim)
    Dg = 0 * np.random.rand(g, dim, dim)

    for i in range(g):
        PSD = sklearn.datasets.make_spd_matrix(dim)
        L = linalg.cholesky(PSD)
        Lg[i] = np.identity(dim) #L
        Dg[i] = np.identity(dim)
    return Lg, Dg

def init_Lg(cluster_Count, p: dim):   # p = dim (Raw Data)
    tensor = np.random.rand(cluster_Count, p, p)
    for cluster in range(cluster_Count):
        A = np.tril(tensor[cluster])
        for k in range(p):
            A[k,k] = 1  # diagonal of ones
        tensor[cluster] = A
    return tensor

def init_Dg(cluster_Count, p: dim): 
    tensorDim = (cluster_Count, p, p)
    tensor = np.ones(tensorDim)
    diag = np.diag(tensor[0])
    for cluster in range(cluster_Count):
        tensor[cluster] = np.diag(diag)
    del diag
    return tensor


if __name__ == '__main__': # test cases
    print(
        init_Z(25, 10),
        init_MixProp(7),
        init_Mean(3,7), 
        init_Psi(5),
        init_Lg(2,3),
        print(init_Lg(3,2)),
        '\n\n\n',
        np.random.rand(2,3,4)[0],
        sep = '\n\n'
    )

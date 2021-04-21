import numpy as np

def subspace_dist(A_1, A_2):
    return np.linalg.norm(A_1 - A_2, ord="fro")

#def spectral_norm(A_1, A_2):
    
    
import numpy as np
from sklearn.decomposition import PCA   


from scipy.stats import wishart



def k_dim_PCA(A, k):
    pca = PCA(n_components=k)
    return pca.fit_transform(A)




# implements Gaussian Mechanism - relasing covariance privately
# as seen from Microsoft paper 
# https://www.microsoft.com/en-us/research/wp-content/uploads/2014/06/PrivatePCA.pdf
def microsoft_alg_1(A, epsilon, delta, k):
    
    gram_A = np.matmul(np.transpose(A), A)
    
    num_data, dimension = A.shape 
    E = np.zeros(gram_A.shape)
    
    
    # set upper triangle including diagonal to normal. set lower triangle
    # to copy of upper triangle
    for row in range(dimension):
        for col in range(row, dimension, 1):
            E[row][col] = np.random.normal(0, 
                                           np.square(delta_1(epsilon, delta)))
            E[col][row] = E[row][col]

    return k_dim_PCA(np.add(gram_A, E), k)
            
# delta function in microsoft for generating symmetric noise
def delta_1(epsilon, delta):
    return np.sqrt(2*np.log(1.25/delta))/epsilon



# implements Laplace input perturbation
# as seen from Jiang, Xie, and Zhang
# https://arxiv.org/pdf/1511.05680.pdf

def laplace_input_perturbation(A, epsilon, k):
    
    gram_A = np.matmul(np.transpose(A), A)
    
    num_data, dimension = A.shape
    L = np.zeros(gram_A.shape)
    
    
    # set upper triangle including diagonal to normal. set lower triangle
    # to copy of upper triangle
    for row in range(dimension):
        for col in range(row, dimension, 1):
            E[row][col] = np.random.laplace(0, 2 * dimension/(num_data * epsilon))

    
    gram_A = gram_A * (1/num_data)
    
    return k_dim_PCA(np.add(gram_A, L), k)




# implements Wishart input perturbation
# https://arxiv.org/pdf/1511.05680.pdf

def wishart_input_perturbation(A, epsilon, k):
    gram_A = np.matmul(np.transpose(A), A)
    num_data, dimension = A.shape
    
    C = np.identity(dimension)*3/(2*num_data*epsilon)
    
    #randomly sample wishart
    W = wishart.rvs(df=dimension+1, scale=C, size=1)
    
    gram_A = gram_A * (1/num_data)
    
    return k_dim_PCA(np.add(gram_A, W), k)


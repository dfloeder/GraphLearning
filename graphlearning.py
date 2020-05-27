'''
GraphPy: Python Module for Graph-based learning algorithms. Efficient implementations of modern methods for graph-based semi-supervised learning, and graph clustering.

See README.md file for usage.

Author: Jeff Calder, 2020
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import scipy.spatial as spatial
import scipy.optimize as opt
import numpy.random as random
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import scipy.sparse.csgraph as csgraph
import sklearn.cluster as cluster
from sklearn.decomposition import PCA
import sys
import time
import csv
import torch

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

#Compute sizes of each class
def label_proportions(labels):
    L = np.unique(labels)
    L = L[L>=0]    

    k = len(L)
    #n = len(labels)
    n = np.sum(labels>=0)
    beta = np.zeros((k,))
    for i in range(k):
        beta[i] = np.sum(labels==L[i])/n

    return beta

#Constructs a weight matrix for graph on mxn grid with NSEW neighbors
def grid_graph(m,n):

    X,Y = np.mgrid[:m,:n]

    return W


#Reweights the graph to use self-tuning weights
def self_tuning(W,D,alpha):
    
    if alpha != 0:
        n = D.shape[0]
        k = D.shape[1]
        d = D[:,k-1]
        d = sparse.spdiags(d**(-alpha),0,n,n)
        W = d*W*d

    return W

#Reweights the graph based on a clustering prior
def cluster_prior(W,cluster_labels):
    
    n = W.shape[0]

    I,J,V = sparse.find(W)
    K = cluster_labels[I] == cluster_labels[J]
    V[K] = V[K]*10
    V = V/np.max(V)

    W = sparse.coo_matrix((V, (I,J)),shape=(n,n)).tocsr()

    return W

#Computes scattering transform of depth 2 of I
#Bruna, Joan, and Stéphane Mallat. "Invariant scattering convolution networks." IEEE transactions on pattern analysis and machine intelligence 35.8 (2013): 1872-1886.
def scattering_transform(I,n,m,depth=2):

    from kymatio import Scattering2D
    print('Computing scattering transform...')

    num_pts = I.shape[0]
    K = torch.from_numpy(I.reshape((num_pts,n,m))).float().contiguous() 
    scattering = Scattering2D(J=depth, shape=(n,m))
    Z = scattering(K).numpy()
    l = Z.shape[1]*Z.shape[2]*Z.shape[3]

    return Z.reshape((num_pts,l))


#Label permutations
#labels = labels
#T = number of trials
#r = label rate in (0,1)
def create_label_permutations_rate(labels,T,R):

    perm = list()
    n = labels.shape[0]
    labelvals = np.unique(labels)
    labelvals = labelvals[labelvals>=0]    
    num_labels = len(labelvals)
    num = np.zeros((num_labels,))
    for i in range(num_labels):
        num[i] = np.sum(labels == labelvals[i])
    
    J = np.arange(n).astype(int)
    for k in range(T):
        for r in R:
            L = []
            for i in range(num_labels):
                l = labelvals[i]
                I = labels==l
                K = J[I]
                m = round(num[i]*r/100)
                L = L + random.choice(K,size=m.astype(int),replace=False).tolist()
            L = np.array(L)
            perm.append(L)

    return perm


#Label permutations
#labels = labels
#T = number of trials
#m = vector of number of labels
def create_label_permutations(labels,T,m,multiplier=None):

    #Find all unique labels >= 0
    #Negative numbers indicate unlabeled nodes
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels>=0]    

    perm = list()
    n = labels.shape[0]
    J = np.arange(n).astype(int)
    for k in range(T):
        for i in m:
            L = []
            ind = 0
            for l in unique_labels:
                I = labels==l
                K = J[I]
                if multiplier is None:
                    L = L + random.choice(K,size=i,replace=False).tolist()
                else:
                    L = L + random.choice(K,size=i*multiplier[ind],replace=False).tolist()
                ind = ind + 1
            L = np.array(L)
            perm.append(L)

    return perm

#Randomly choose m labels per class
def randomize_labels(L,m):

    perm = create_label_permutations(L,1,[m])

    return perm[0]

#Default function
def exp_weight(x):
    return np.exp(-x)

#Compute degrees of weight matrix W
def degrees(W):

    return np.squeeze(np.array(np.sum(W,axis=1)))


#Multiply diagonal of matrix by degree
def diag_multiply(W,b):

    n = W.shape[0]  #Number of points

    D = sparse.spdiags(W.diagonal(),0,n,n)
    
    return W - (1-b)*D

#Compute degrees of weight matrix W
#Returns sparse matrix with degrees on diagonal
def degree_matrix(W,p=1):

    n = W.shape[0]  #Number of points

    #Construct sparse degree matrix
    d = degrees(W)
    D = sparse.spdiags(d**p,0,n,n)

    return D.tocsr()

#Laplace matrix
#W = weight matrix
#norm = type of normalization
#   Options: none, randomwalk, normalized
def graph_laplacian(W,norm="none"):

    D = degree_matrix(W)

    if norm=="none":
        L = D - W
    elif norm=="randomwalk1":
        Dinv = degree_matrix(W,p=-1)
        L = Dinv*(D-W)
    elif norm=="randomwalk2":
        Dinv = degree_matrix(W,p=-1)
        L = (D-W)*Dinv
    elif norm=="normalized":
        Dinv2 = degree_matrix(W,p=-1/2)
        L = Dinv2*(D-W)*Dinv2
    else:
        print("Invalid option for graph Laplacian normalization. Returning unnormalized Laplacian.")
        L = D - W

    return L.tocsr()

#Graph infinity Laplacian
#W = sparse weight matrix
#u = function on graph
def graph_phi_laplacian(W,u,phi,I=None,J=None,V=None):

    n = W.shape[0]
    if I is None or J is None:
        I,J,V = sparse.find(W)

    w = u[J]-u[I]
    a = np.absolute(w)
    pa = phi(a)
    m = pa/(a+1e-13)
    M = sparse.coo_matrix((V*pa/(a+1e-13), (I,J)),shape=(n,n)).tocsr()
    m = degrees(M)

    M = sparse.coo_matrix((V*pa*np.sign(w), (I,J)),shape=(n,n)).tocsr()
    M = np.squeeze(np.array(np.sum(M,axis=1)))

    return M, m


#Graph infinity Laplacian
#W = sparse weight matrix
#u = function on graph
def graph_infinity_laplacian(W,u,I=None,J=None,V=None):

    n = W.shape[0]
    if I is None or J is None:
        I,J,V = sparse.find(W)
    M = sparse.coo_matrix((V*(u[J]-u[I]), (I,J)),shape=(n,n)).tocsr()
    M = M.min(axis=1) + M.max(axis=1)

    return M.toarray().flatten()


#Construct epsilon-graph sparse distance matrix
def eps_weight_matrix(X,eps,f=exp_weight):

    n = X.shape[0]  #Number of points

    #Rangesearch to find nearest neighbors
    Xtree = spatial.cKDTree(X)
    M = Xtree.query_pairs(eps)
    M = np.array(list(M))

    #Differences between points and neighbors
    V = X[M[:,0],:] - X[M[:,1],:]
    D = np.sum(V*V,axis=1)

    #Weights
    D = f(4*D/(eps*eps))

    #Symmetrize weights and add diagonal entries
    D = np.concatenate((D,D,f(0)*np.ones(n,)))
    M1 = np.concatenate((M[:,0],M[:,1],np.arange(0,n)))
    M2 = np.concatenate((M[:,1],M[:,0],np.arange(0,n)))

    #Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    W = sparse.coo_matrix((D, (M1,M2)),shape=(n,n))

    return W.tocsr()

#Exact knnsearch
def knnsearch(X,k):
    #KDtree to find nearest neighbors
    n = X.shape[0]
    Xtree = spatial.cKDTree(X)
    D, J = Xtree.query(X,k=k)
    I = np.ones((n,k))*J[:,0][:,None]

    return I,J,D

#Perform approximate nearest neighbor search, returning indices I,J of neighbors, and distance D
# Metric can be "angular", "euclidean", "manhattan", "hamming", or "dot".
def knnsearch_annoy(X,k, similarity='euclidean'):

    from annoy import AnnoyIndex

    n = X.shape[0]  #Number of points
    dim = X.shape[1]#Dimension

    print('kNN search with Annoy approximate nearest neighbor package...')
    printProgressBar(0, n, prefix = 'Progress:', suffix = 'Complete', length = 50)

    u = AnnoyIndex(dim, similarity)  # Length of item vector that will be indexed
    for i in range(n):
        u.add_item(i, X[i,:])

    u.build(10)  #10 trees
    
    D = []
    I = []
    J = []
    for i in range(n):
        printProgressBar(i+1, n, prefix = 'Progress:', suffix = 'Complete', length = 50)
        A = u.get_nns_by_item(i, k,include_distances=True,search_k=-1)
        I.append([i]*k)
        J.append(A[0])
        D.append(A[1])

    I = np.array(I)
    J = np.array(J)
    D = np.array(D)

    return I,J,D

#Compute weight matrix from nearest neighbor indices I,J and distances D
def weight_matrix_selftuning(I,J,D):

    n = I.shape[0]
    k = I.shape[1]

    #Distance to kth nearest neighbor as a matrix
    sigma = D[:,k-1]
    sigma = sparse.spdiags(1/sigma,0,n,n)
    sigma = sigma.tocsr()

    #Flatten
    I = I.flatten()
    J = J.flatten()
    D = D.flatten()

    #Symmetrize and remove redundant entries
    M1 = np.vstack((I,J,D))
    M2 = np.vstack((J,I,D))
    M = np.concatenate((M1,M2),axis=1)
    M = np.unique(M,axis=1)

    #Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    I = M[0,:]
    J = M[1,:]
    D = M[2,:]
    dist = sparse.coo_matrix((D,(I,J)),shape=(n,n)).tocsr()
    B = sparse.coo_matrix((np.ones(len(D),),(I,J)),shape=(n,n)).tocsr() #Ones in all entries

    #Self-tuning weights
    E = -4*sigma*(dist**2)*sigma
    W = E.expm1()
    W = W.multiply(B) + B

    return W

#Compute weight matrix from nearest neighbor indices I,J and distances D
#k = number of neighbors
#Chooses k neighbors at random from I.shape[1] nearset neighbors
def weight_matrix_homogenized(I,J,D,k,f=exp_weight):

    #I = I[:,:10]
    #J = J[:,:10]
    #D = D[:,:10]

    #Restrict I,J,D to k neighbors
    k = np.minimum(I.shape[1],k)
    n = I.shape[0]
    for i in range(n):
        ind = random.choice(I.shape[1],k,replace=False)
        I[i,:k] = I[i,ind]
        J[i,:k] = J[i,ind]
        D[i,:k] = 1

    n = I.shape[0]
    k = I.shape[1]

    D = D*D
    eps = D[:,k-1]/4
    D = f(D/eps[:,None])

    #Flatten
    I = I.flatten()
    J = J.flatten()
    D = D.flatten()

    #Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    W = sparse.coo_matrix((D, (I,J)),shape=(n,n)).tocsr()

    return W

#Compute distance matrix from nearest neighbor indices I,J and distances D
#k = number of neighbors
def dist_matrix(I,J,D,k):

    #Restrict I,J,D to k neighbors
    k = np.minimum(I.shape[1],k)
    I = I[:,:k]
    J = J[:,:k]
    D = D[:,:k]

    n = I.shape[0]
    k = I.shape[1]

    #Flatten
    I = I.flatten()
    J = J.flatten()
    D = D.flatten()

    #Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    W = sparse.coo_matrix((D, (I,J)),shape=(n,n)).tocsr()

    return W

#Adds weights to an adjacency matrix W using similarity in data X
def add_weights(W,X,labels):

    n = W.shape[0]

    #pca = PCA(n_components=20)
    #X = pca.fit_transform(X)
    #print(X.shape)

    I,J,V = sparse.find(W)
    
    #Dot products
    Y = X[I,:]-X[J,:]
    Y = np.sum(Y*Y,axis=1)

    W = sparse.coo_matrix((Y, (I,J)),shape=(n,n)).tocsr()
    max_dist = np.reshape(np.max(W,axis=1).todense().tolist(),(n,))
    D = sparse.spdiags((max_dist+1e-10)**(-1),0,n,n).tocsr()
    W = D*W

    I,J,V = sparse.find(W)
    V = np.exp(-2*V)
    W = sparse.coo_matrix((V, (I,J)),shape=(n,n)).tocsr()

    return W

#Finds largest connected component of the graph represented by adjacency matrix W
#Returns the weighted adjacency matrix, along with a boolean mask indicating the 
#vertices from the input matrix that were selected
def largest_conn_component(W):

    ncomp,labels = csgraph.connected_components(W,directed=False) 
    num_verts = np.zeros((ncomp,))
    for i in range(ncomp):
        num_verts[i] = np.sum(labels==i)
    
    i_max = np.argmax(num_verts)
    ind = labels==i_max

    A = W[ind,:]
    A = A[:,ind]

    print("Found %d"%ncomp+" connected components.")
    print("Returning component with %d"%num_verts[i_max]+" vertices out of %d"%W.shape[0]+" total vertices.")

    return A,ind

#Compute weight matrix from nearest neighbor indices I,J and distances D
#k = number of neighbors
def weight_matrix(I,J,D,k,f=exp_weight,symmetrize=True):

    #Restrict I,J,D to k neighbors
    k = np.minimum(I.shape[1],k)
    I = I[:,:k]
    J = J[:,:k]
    D = D[:,:k]

    n = I.shape[0]
    k = I.shape[1]

    D = D*D
    eps = D[:,k-1]/4
    D = f(D/eps[:,None])

    #Flatten
    I = I.flatten()
    J = J.flatten()
    D = D.flatten()

    #Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    W = sparse.coo_matrix((D, (I,J)),shape=(n,n)).tocsr()

    if symmetrize:
        W = (W + W.transpose())/2;

    return W

#Construct k-nn sparse distance matrix
#Note: Matrix is not symmetric
def knn_weight_matrix(X,k,f=exp_weight):

    I,J,D = knnsearch_annoy(X,k)
    W = weight_matrix(I,J,D,k,f=f)
   
    return W

#Poisson solve
#Solves Lu = f with preconditioned conjugate gradient
def pcg_solve(L,f,x0=None,tol=1e-10):

    #start_time = time.time()
    L = L.tocsr()

    #Conjugate gradient with Jacobi preconditioner
    m = L.shape[0]
    M = L.diagonal()
    M = sparse.spdiags(1/M,0,m,m).tocsr()
    if x0 is None:
        u,i = splinalg.cg(L,f,tol=tol,M=M)
    else:
        u,i = splinalg.cg(L,f,x0=x0,tol=tol,M=M)
    #print("--- %s seconds ---" % (time.time() - start_time))

    return u

#Constrained linear solve
#Solves Lu = f subject to u(I)=g
def constrained_solve(L,I,g,f=None,x0=None,tol=1e-10):

    L = L.tocsr()
    n = L.shape[0]

    #Locations of labels
    idx = np.full((n,), True, dtype=bool)
    idx[I] = False

    #Right hand side
    b = -L[:,I]*g
    b = b[idx]

    if f is not None:
        b = b + f[idx]

    #Left hand side matrix
    A = L[idx,:]
    A = A[:,idx]
    
    #start_time = time.time()

    #Conjugate gradient with Jacobi preconditioner
    m = A.shape[0]
    M = A.diagonal()
    M = sparse.spdiags(1/(M+1e-10),0,m,m).tocsr()

    if x0 is None:
        v,i = splinalg.cg(A,b,tol=tol,M=M)
    else:
        v,i = splinalg.cg(A,b,x0=x0[idx],tol=tol,M=M)
    #print("--- %s seconds ---" % (time.time() - start_time))

    #Add labels back into array
    u = np.ones((n,))
    u[idx] = v
    u[I] = g

    return u

#Returns n random points in R^d
def rand(n,d):
    return random.rand(n,d)

def randn(n,d):
    X = np.zeros((n,d))
    for i in range(d):
        X[:,i] = np.random.normal(0,1,n) 

    return X

def bean_data(n,h):

    #n = number of points
    #h = height of bridge (h=0.2)

    a=-1
    b=1
    x = a + (b-a)*random.rand(3*n);
    c=-0.6
    d=0.6;
    y = c + (d-c)*random.rand(3*n);

    X=np.transpose(np.vstack((x,y)))

    dist_from_x_axis=0.4*np.sqrt(1-x**2)*(1+h-np.cos(3*x))
    in_bean = abs(y) <= dist_from_x_axis
    X = X[in_bean,:]
    if X.shape[0] < n:
        print('Not enough samples');
    else:
        X = X[:n,:]

    return X


#Triangulation of domain
def mesh(X):

    n = X.shape[0]
    d = X.shape[1]
    if d > 2:
        X = X[:,0:2]

    #Normalize data to unit box
    x1 = X[:,0].min()
    x2 = X[:,0].max()
    y1 = X[:,1].min()
    y2 = X[:,1].max()
    X = X - [x1,y1]
    X[:,0] = X[:,0]/(x2-x1)
    X[:,1] = X[:,1]/(y2-y1)

    #Add padding data around
    pad = 10/np.sqrt(n)
    m = int(pad*n)
    Y = rand(m,2)
    Y[:,0] = Y[:,0]*pad - pad
    Z = np.vstack((X,Y))
    Y = rand(m,2)
    Y[:,0] = Y[:,0]*pad + 1
    Z = np.vstack((Z,Y))
    Y = rand(m,2)
    Y[:,1] = Y[:,1]*pad - pad
    Z = np.vstack((Z,Y))
    Y = rand(m,2)
    Y[:,1] = Y[:,1]*pad + 1
    Z = np.vstack((Z,Y))

    #Delaunay triangulation
    T = spatial.Delaunay(Z);
    Tri = T.simplices
    J = np.sum(Tri >= n,axis=1)==0;
    Tri = Tri[J,:]

    return Tri

#def plot(X,u):
#    Tri = mesh(X)
#
#    import mayavi.mlab as mlab
#    mlab.triangular_mesh(X[:,0],X[:,1],u,Tri)
#    mlab.view(azimuth=-45,elevation=60)

#Laplace learning
#Zhu, Xiaojin, Zoubin Ghahramani, and John D. Lafferty. "Semi-supervised learning using gaussian fields and harmonic functions." Proceedings of the 20th International conference on Machine learning (ICML-03). 2003.
def laplace_solve(W,I,g,norm="none"):
    L = graph_laplacian(W,norm=norm)
    return constrained_solve(L,I,g)

#Shift trick
#W = Weight matrix
#I = indices of labels
#g = +1/-1 values of labels
def shift_solve(W,I,g):
    
    #Laplace learning
    u = laplace_solve(W,I,g)

    #Shift solution
    s = degrees(W)
    c = np.sum(s[I]*g)/sum(s[I])
    u = u - c 
    
    u = u - np.mean(u)

    return u


#Shift trick by mean
#W = Weight matrix
#I = indices of labels
#g = +1/-1 values of labels
def meanshift_solve(W,I,g):
    
    #Laplace learning
    u = laplace_solve(W,I,g)

    #Center solution
    u = u - np.mean(u)

    return u

#Reweights the weight matrix for WNLL
def wnll(W,I):

    n = W.shape[0]
    m = len(I)

    a = np.ones((n,))
    a[I] = n/m
    
    D = sparse.spdiags(a,0,n,n).tocsr()
    W = D*W + W*D

    return W


#Weighted nonlocal Laplacian
#Shi, Zuoqiang, Stanley Osher, and Wei Zhu. "Weighted nonlocal laplacian on interpolation from sparse data." Journal of Scientific Computing 73.2-3 (2017): 1164-1177.
def wnll_solve(W,I,g):

    n = W.shape[0]
    W = wnll(W,I)
    L = graph_laplacian(W,norm="none")
    return constrained_solve(L,I,g)

#Properly weighted Laplacian
#Calder, Jeff, and Dejan Slepcev. "Properly-weighted graph Laplacian for semi-supervised learning." arXiv preprint arXiv:1810.04351 (2018).
def properlyweighted_solve(W,I,g,X,alpha,zeta,r):

    n = W.shape[0]
    rzeta = r/(zeta-1)**(1/alpha)

    Xtree = spatial.cKDTree(X[I,:])
    D, J = Xtree.query(X)
    D[D < rzeta] = rzeta
    gamma = 1 + (r/D)**alpha

    D = sparse.spdiags(gamma,0,n,n).tocsr()

    L = graph_laplacian(D*W + W*D,norm="none")

    return constrained_solve(L,I,g)

#Game theoretic p-Laplace learning
#Rios, Mauricio Flores, Jeff Calder, and Gilad Lerman. "Algorithms for $\ell_p$-based semi-supervised learning on graphs." arXiv preprint arXiv:1901.05031 (2019).
def plaplace_solve(W,I,g,p,sol_method="SemiImplicit",norm="none"):

    #start_time = time.time()

    n = W.shape[0]
    W = W/W.max()
    
    if p == float("inf"):
        alpha = 0
        delta = 1
    else:
        alpha = 1/p
        delta = 1-2/p
    
    dx = degrees(W)
    theta = 1.2*(2*alpha + np.max(dx)*delta)

    if p == float("inf"):
        beta = 1
        gamma = 1/theta
    else:
        beta = (theta*p - 2)/(theta*p)
        gamma = (p-2)/(theta*p-2)

    if norm=="normalized":
        deg = dx[I]**(1/2) 
        g = g/deg

    L = graph_laplacian(W)
    u = constrained_solve(L,I,g)
    uu = np.max(g)*np.ones((n,))
    ul = np.min(g)*np.ones((n,))

    WI,WJ,WV = sparse.find(W)

    #Set labels
    u[I] = g
    uu[I] = g
    ul[I] = g

    #Time step for gradient descent
    dt = 0.9/(alpha + 2*delta)

    if sol_method=="GradientDescentCcode":
        try:
            import cmodules.cgraphpy as cgp
        except:
            print("cgraphpy cmodule not found. You may just need to compile it.")
            sys.exit()

        #Type casting and memory blocking
        uu = np.ascontiguousarray(uu,dtype=np.float64)
        ul = np.ascontiguousarray(ul,dtype=np.float64)
        WI = np.ascontiguousarray(WI,dtype=np.int32)
        WJ = np.ascontiguousarray(WJ,dtype=np.int32)
        WV = np.ascontiguousarray(WV,dtype=np.float64)
        I = np.ascontiguousarray(I,dtype=np.int32)
        g = np.ascontiguousarray(g,dtype=np.float64)

        cgp.lp_iterate(uu,ul,WI,WJ,WV,I,g,p,1e6,1e-1,0.0)
        u = (uu+ul)/2

        #Check residual
        L2uu = -L*uu
        LIuu = graph_infinity_laplacian(W,uu,I=WI,J=WJ,V=WV)
        resu = alpha*L2uu/dx + delta*LIuu
        resu[I]=0

        L2ul = -L*ul
        LIul = graph_infinity_laplacian(W,ul,I=WI,J=WJ,V=WV)
        resl = alpha*L2ul/dx + delta*LIul
        resl[I]=0

        #print('Upper residual = %f' % np.max(np.absolute(resu)))
        #print('Lower residual = %f' % np.max(np.absolute(resl)))

    else:
        err = 1e6
        i = 0
        while err > 1e-1:

            i+=1
            
            #Graph laplacians
            L2u = -L*u
            LIu = graph_infinity_laplacian(W,u,I=WI,J=WJ,V=WV)

            #Residual error
            res = alpha*L2u/dx + delta*LIu
            res[I]=0
            #err = np.max(np.absolute(res))
            #print("Residual error = "+str(err))

            #Update
            if sol_method=="GradientDescent":
                L2uu = -L*uu
                LIuu = graph_infinity_laplacian(W,uu,I=WI,J=WJ,V=WV)
                res = alpha*L2uu/dx + delta*LIuu
                res[I]=0
                uu = uu + dt*res        
                err = np.max(np.absolute(res))
                #print("Upper residual = "+str(err))

                L2ul = -L*ul
                LIul = graph_infinity_laplacian(W,ul,I=WI,J=WJ,V=WV)
                res = alpha*L2ul/dx + delta*LIul
                res[I]=0
                ul = ul + dt*res        
                err = np.max(np.absolute(res))
                #print("Lower residual = "+str(err))
                err1 = np.max(uu-ul)
                err2 = np.min(uu-ul)

                #print("Residual error = "+str(err1)+","+str(err2))
                err = err1

                u = (uu + ul)/2
            elif sol_method=="SemiImplicit":
                rhs = beta*(2*gamma*dx*LIu - L2u)
                u = constrained_solve(L,I,g,f=rhs,x0=u,tol=err/100)
            else:
                print("Invalid p-Laplace solution method.")
                sys.exit()
            
    if norm=="normalized":
        deg = dx**(1/2) 
        u = u*deg

    #print("--- %s seconds ---" % (time.time() - start_time))
    return u

#Gradient of function on graph
#W = sparse weight matrix
#u = function on graph
def graph_gradient(W,u,I=None,J=None,V=None):

    n = W.shape[0]
    if I is None or J is None:
        I,J,V = sparse.find(W)

    G = sparse.coo_matrix((V*(u[J]-u[I]), (I,J)),shape=(n,n)).tocsr()

    return G

#Divergence of vector field F (F should be skew-symmetric)
#F = sparse matrix representing vector field
def graph_divergence(F,W):
    
    F = F.multiply(W)
    return 2*np.squeeze(np.array(np.sum(F,axis=1)))


#Random-walk SSL 
#Zhou, Dengyong, et al. "Learning with local and global consistency." Advances in neural information processing systems. 2004.
def randomwalk_solve(W,I,g,epsilon):
    
    n = W.shape[0]

    #Zero diagonals
    W = W - sparse.spdiags(W.diagonal(),0,n,n)

    #Construct Laplacian matrix
    Dinv2 = degree_matrix(W,p=-1/2)
    L = sparse.identity(n) - (1-epsilon)*Dinv2*W*Dinv2;

    #Format right hand side
    b = np.zeros((n,))
    b[I] = g

    return pcg_solve(L,b)

#Computes accuracy of labeling
#m = number of labeled points used
def accuracy(L,L_true,m):   
    #Remove unlabeled nodes
    I = L_true >=0
    L = L[I]
    L_true = L_true[I]

    #Compute accuracy
    return 100*np.maximum(np.sum(L==L_true)-m,0)/(len(L)-m)

#Projects all columns of (kxn) matrix X onto k-simplex
def ProjectToSimplex(X):
   
    n = X.shape[1]
    k = X.shape[0]

    Xs = -np.sort(-X,axis=0)  #Sort descending
    A = np.tril(np.ones((k,k)))
    Sum = A@Xs
    Max = np.transpose((np.transpose(Sum) - 1)/(np.arange(k)+1))
    Xs[:-1,:] = Xs[1:,:]
    Xs[-1,:] = (Sum[k-1,:]-1)/k
    I = np.argmax(Max >= Xs,axis=0)
    X = np.maximum(X-Max[I,range(n)],0)
    return X

#Takes list of labels and converts to vertices of simplex format
def LabelsToVec(L):

    n = L.shape[0]

    labels = np.unique(L)
    k = len(labels)
    for i in range(k):
        L[L==labels[i]] = i

    L = L.astype(int)
    X = np.zeros((k,n))
    X[L,range(n)] = 1

    return X,labels

#Projects all rows of (nxk) matrix X to closest vertex of the simplex
#Assume X already lives in the simplex, e.g., is the output of ProjectToSimplex
def ClosestVertex(X):

    n = X.shape[1]
    k = X.shape[0]
    L = np.argmax(X,axis=0)
    X = np.zeros((k,n))
    X[L,range(n)] = 1
    return X

#Threshold with temperature to closest vertex
def ClosestVertexTemp(X,T=0.01):

    n = X.shape[1]
    k = X.shape[0]
    
    beta = 1/T
    Y = np.exp(beta*X)
    Ysum = np.sum(Y,axis=0)
    Y = Y/Ysum

    X[0,:] = Y[0,:]
    for i in range(1,k):
        X[i,:] = X[i-1,:] + Y[i,:]

    R = random.rand(n,1)
    L = np.sum(R.flatten() > X,axis=0)

    X = np.zeros((k,n))
    X[L,range(n)] = 1
    return X

#Volume MBO, initialized with Poisson
def poisson_volumeMBO(W,I,g,dataset,beta,T):

    #Set diagonal entries to zero
    W = diag_multiply(W,0)

    try:
        import cmodules.cgraphpy as cgp
    except:
        print("cgraphpy cmodule not found. You may just need to compile it.")
        sys.exit()

    #Solve Poisson problem and compute labels
    u,_ = poisson(W,I,g)
    max_locations = np.argmax(u,axis=0)
    u = (np.unique(g))[max_locations]

    n = W.shape[0]
    k = len(np.unique(g))
    WI,WJ,WV = sparse.find(W)

    #Class counts
    ClassCounts = (n*beta).astype(int)

    #Type casting and memory blocking
    u = np.ascontiguousarray(u,dtype=np.int32)
    WI = np.ascontiguousarray(WI,dtype=np.int32)
    WJ = np.ascontiguousarray(WJ,dtype=np.int32)
    WV = np.ascontiguousarray(WV,dtype=np.float32)
    I = np.ascontiguousarray(I,dtype=np.int32)
    g = np.ascontiguousarray(g,dtype=np.int32)
    ClassCounts = np.ascontiguousarray(ClassCounts,dtype=np.int32)

    cgp.volume_mbo(u,WI,WJ,WV,I,g,ClassCounts,k,0.0,T)

    #Set given labels and convert to vector format
    u[I] = g
    u,_ = LabelsToVec(u)
    return u



#Volume MBO (Jacobs, et al.)
def volumeMBO(W,I,g,dataset,beta,T):

    #Set diagonal entries to zero
    W = diag_multiply(W,0)

    try:
        import cmodules.cgraphpy as cgp
    except:
        print("cgraphpy cmodule not found. You may just need to compile it.")
        sys.exit()

    n = W.shape[0]
    k = len(np.unique(g))
    u = np.zeros((n,))
    WI,WJ,WV = sparse.find(W)

    #Class counts
    ClassCounts = (n*beta).astype(int)

    #Type casting and memory blocking
    u = np.ascontiguousarray(u,dtype=np.int32)
    WI = np.ascontiguousarray(WI,dtype=np.int32)
    WJ = np.ascontiguousarray(WJ,dtype=np.int32)
    WV = np.ascontiguousarray(WV,dtype=np.float32)
    I = np.ascontiguousarray(I,dtype=np.int32)
    g = np.ascontiguousarray(g,dtype=np.int32)
    ClassCounts = np.ascontiguousarray(ClassCounts,dtype=np.int32)

    cgp.volume_mbo(u,WI,WJ,WV,I,g,ClassCounts,k,1.0,T)

    #Set given labels and convert to vector format
    u[I] = g
    u,_ = LabelsToVec(u)
    return u


#Multiclass MBO
#Garcia-Cardona, Cristina, et al. "Multiclass data segmentation using diffuse interface methods on graphs." IEEE transactions on pattern analysis and machine intelligence 36.8 (2014): 1600-1613.
def multiclassMBO(W,I,g,eigvals,eigvecs,dataset,true_labels=None):

    n = W.shape[0]
    k = len(np.unique(g))

    Ns = 6
    if dataset=='MNIST' or dataset=='FashionMNIST' or dataset=='cifar':
        dt = 0.15
        mu = 50
    elif dataset=='WEBKB':
        dt = 1
        mu = 4
    else:
        print('Dataset not supported by MBO...')
        sys.exit(2)

    #Load eigenvalues and eigenvectors
    X = eigvecs
    num_eig = len(eigvals)
    
    #Form matrices
    V = np.diag(1/(1 + (dt/Ns)*eigvals)) 
    Y = X@V
    Xt = np.transpose(X)

    #Random initial labeling
    u = random.rand(k,n)
    u = ProjectToSimplex(u)

    #Set initial known labels
    J = np.zeros(n,)
    K = np.ones(n,)*g[0]
    J[I] = 1
    K[I] = g
    Kg,_ = LabelsToVec(K)
    Kg = Kg*J
    u = Kg + (1-J)*u
    
    #Maximum number of iterations
    T = 10
    for i in range(T):
        for s in range(Ns):
            Z = (u - (dt/Ns)*mu*J*(u - Kg))@Y
            u = Z@Xt
            
        #Projection step
        u = ProjectToSimplex(u)
        u = ClosestVertex(u)

        #Compute accuracy if all labels are provided
        if true_labels is not None:
            max_locations = np.argmax(u,axis=0)
            labels = (np.unique(g))[max_locations]
            labels[I] = g
            acc = accuracy(labels,true_labels,len(I))
            print('Accuracy = %.2f'%acc)

    return u


#Poisson MBO
def poissonMBO(W,I,g,dataset,beta,true_labels=None,temp=0,use_cuda=False):

    n = W.shape[0]
    k = len(np.unique(g))

    W = diag_multiply(W,0)
    if dataset=='MNIST' or dataset=='FashionMNIST':
        mu = 1
        Ns = 50
    elif dataset=='WEBKB':
        mu = 1000
        Ns = 5
    elif dataset=='citeseer':
        mu = 1
        Ns = 500
    elif dataset=='cifar':
        mu = 1
        Ns = 30
    else:
        print('Dataset not supported by MBO...')
        sys.exit(2)

    
    #Labels to vector and correct position
    J = np.zeros(n,)
    K = np.ones(n,)*g[0]
    J[I] = 1
    K[I] = g
    Kg,_ = LabelsToVec(K)
    Kg = Kg*J
    
    #Poisson source term
    c = np.sum(Kg,axis=1)/len(I)
    b = np.transpose(Kg)
    b[I,:] = b[I,:]-c
    b = np.transpose(b)

    L = graph_laplacian(W,norm='none')

    #Initialize u via Poisson learning
    #u = np.zeros((k,n))
    #for j in range(k):
    #    u[j,:] = pcg_solve(L,b[j,:])
    #u = mu*u
    #u = np.transpose(np.transpose(u) - np.mean(u,axis=1))
    u,mix_time = poisson(W,I,g,use_cuda=use_cuda)
    #Ns = int(mix_time/4)
    u = ProjectToSimplex(u)
    u = ClosestVertex(u)

    #Time step for stability
    dt = 1/np.max(degrees(W))

    P = sparse.identity(n) - dt*L
    Db = mu*dt*b

    if use_cuda:
        Pt = torch_sparse(P).cuda()
        Dbt = torch.from_numpy(np.transpose(Db)).float().cuda()

    T = 10 #Maximum number of iterations
    for i in range(T):

        if use_cuda:

            #Put on GPU and run heat equation
            ut = torch.from_numpy(np.transpose(u)).float().cuda()
            for s in range(Ns):
                #u = u*P + Db
                ut = torch.sparse.addmm(Dbt,Pt,ut)

            #Put back on CPU
            u = np.transpose(ut.cpu().numpy())
         
        else: #Use CPU 
            for s in range(Ns):
                #u = u + dt*(mu*b - u*L)
                u = u*P + Db

        #Projection step
        u = ProjectToSimplex(u)
        u = ClosestVertex(u)
        u = np.transpose(np.transpose(u) - np.mean(u,axis=1) + beta)

        #Compute accuracy if all labels are provided
        if true_labels is not None:
            max_locations = np.argmax(u,axis=0)
            labels = (np.unique(g))[max_locations]
            labels[I] = g
            acc = accuracy(labels,true_labels,len(I))
            print('Accuracy = %.2f'%acc)
    
    return u

def torch_sparse(A):

    A = A.tocoo()
    values = A.data
    indices = np.vstack((A.row, A.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

#Sparse Label Propagation
def SparseLabelPropagation(W,I,g,true_labels=None):

    n = W.shape[0]
    k = len(np.unique(g))

    WI,WJ,WV = sparse.find(W)
    B = sparse.coo_matrix((np.ones(len(WV),),(WI,WJ)),shape=(n,n)).tocsr() #Ones in all entries

    #Construct matrix 1/2W and 1/deg
    lam = 2*W - (1-1e-10)*B
    lam = -lam.log1p()
    lam = lam.expm1() + B
    Id = sparse.identity(n) 
    gamma = degree_matrix(W+1e-10*Id,p=-1)

    #Random initial labeling
    #u = random.rand(k,n)
    u = np.zeros((k,n))

    #Set initial known labels
    J = np.zeros(n,)
    K = np.ones(n,)*g[0]
    J[I] = 1
    K[I] = g
    Kg,_ = LabelsToVec(K)
    Kg = Kg*J

    #Initialization
    Y = list()
    for j in range(k):
        Gu = graph_gradient(W,u[j,:],I=WI,J=WJ,V=WV)
        Y.append(Gu)

    #Main loop for sparse label propagation
    T = 100
    for i in range(T):

        u_prev = np.copy(u)
        #Compute div
        for j in range(k):
            div = graph_divergence(Y[j],W)
            u[j,:] = u_prev[j,:] - gamma*div
            u[j,I] = Kg[j,I]  #Set labels
            u_tilde = 2*u[j,:] - u_prev[j,:]

            Gu = -graph_gradient(W,u_tilde,I=WI,J=WJ,V=WV)
            Y[j] = Y[j] + Gu.multiply(lam)
            ind1 = B.multiply(abs(Y[j])>1)
            ind2 = B - ind1
            Y[j] = ind1.multiply(Y[j].sign()) + ind2.multiply(Y[j])

        #Compute accuracy if all labels are provided
        if true_labels is not None:
            max_locations = np.argmax(u,axis=0)
            labels = (np.unique(g))[max_locations]
            labels[I] = g
            acc = accuracy(labels,true_labels,len(I))
            print('Accuracy = %.2f'%acc)

    return u


#Dynamic Label Propagation
def DynamicLabelPropagation(W,I,g,alpha=0.05,lam=0.1,true_labels=None):

    n = W.shape[0]
    k = len(np.unique(g))

    W = diag_multiply(W,0)
    
    #Labels to vector and correct position
    J = np.zeros(n,)
    K = np.ones(n,)*g[0]
    J[I] = 1
    K[I] = g
    u,_ = LabelsToVec(K)
    u = u*J

    #Set initial known labels
    J = np.zeros(n,)
    K = np.ones(n,)*g[0]
    J[I] = 1
    K[I] = g
    Kg,_ = LabelsToVec(K)
    Kg = np.transpose(Kg*J)
    u = np.copy(Kg)
  
    if n > 5000:
        print("Cannot use Dynamic Label Propagation on large datasets.")
    else:
        #Setup matrices
        Id = sparse.identity(n) 
        D = degree_matrix(W,p=-1)
        P = D*W
        P = np.array(P.todense())
        Pt = np.copy(P)

        T = 2
        for i in range(T):
            v = P@u
            u = Pt@u
            u[I,:] = Kg[I,:]
            Pt = P@Pt@np.transpose(P) + alpha*v@np.transpose(v) + lam*Id

            #Compute accuracy if all labels are provided
            if true_labels is not None:
                u = np.array(u)
                max_locations = np.argmax(u,axis=1)
                labels = (np.unique(g))[max_locations]
                labels[I] = g
                acc = accuracy(labels,true_labels,len(I))
                print('i:%d'%i+',Accuracy = %.2f'%acc)
        

        u = np.transpose(np.array(u))

    return u

#Centered and Iterated Centered Kernel of Mai/Coulliet 2018 
def CenteredKernel(W,I,g,true_labels=None):

    n = W.shape[0]
    k = len(np.unique(g))

    #W = diag_multiply(W,0)
    
    #Labels to vector and correct position
    J = np.zeros(n,)
    K = np.ones(n,)*g[0]
    J[I] = 1
    K[I] = g
    Kg,_ = LabelsToVec(K)
    Kg = np.transpose(Kg*J)
    
    u = np.copy(Kg)
    v = np.ones((n,1))
    vt = np.ones((1,n))

    e = np.random.rand(n,1)
    for i in range(100):
        y = W*(e -  (1/n)*v@(vt@e))
        w = y - (1/n)*v@(vt@y) #=Ae
        l = abs(np.transpose(e)@w/(np.transpose(e)@e))
        e = w/np.linalg.norm(w)

    #Number of iterations
    alpha = 5*l/4
    T = 1000
    err = 1
    while err > 1e-10:
        y = W*(u -  (1/n)*v@(vt@u))
        w = (1/alpha)*(y - (1/n)*v@(vt@y)) - u #Laplacian
        w[I,:] = 0
        err = np.max(np.absolute(w))
        u = u + w

        #Compute accuracy if all labels are provided
        if true_labels is not None:
            max_locations = np.argmax(u,axis=1)
            labels = (np.unique(g))[max_locations]
            labels[I] = g
            acc = accuracy(labels,true_labels,len(I))
            print('Accuracy = %.2f'%acc)
    
    return np.transpose(u)





#Poisson learning
def poisson(W,I,g,true_labels=None,use_cuda=False):

    n = W.shape[0]
    unique_labels = np.unique(g)
    k = len(unique_labels)
    
    num_labels = np.zeros((k,))
    for i in range(k):
        num_labels[i] = np.sum(g==unique_labels[i])

    multiplier = 1/num_labels

    W = diag_multiply(W,0)
    
    #Labels to vector and correct position
    J = np.zeros(n,)
    K = np.ones(n,)*g[0]
    J[I] = 1
    K[I] = g
    Kg,_ = LabelsToVec(K)
    Kg = Kg*J
    Kg = np.transpose(multiplier*np.transpose(Kg))
    
    #Poisson source term
    #c = np.sum(Kg,axis=1)/len(I)
    c = np.sum(Kg,axis=1)/k
    b = np.transpose(Kg)
    #b[I,:] = b[I,:]-c/num_labels
    j=0
    for i in I:
        b[i,:] = b[i,] - c/num_labels[g[j]]
        j = j+1

    #Setup matrices
    L = graph_laplacian(W,norm='none')
    D = degree_matrix(W + 1e-10*sparse.identity(n),p=-1)
    P = sparse.identity(n) - D*L
    Db = D*b

    v = np.max(Kg,axis=0)
    v = v/np.sum(v)
    vinf = degrees(W)/np.sum(degrees(W))
    RW = W*D
    u = np.zeros((n,k))

    #Number of iterations
    #T = int(n*2/700) #OLD number iterations
    T = 0
    if use_cuda:
        
        Pt = torch_sparse(P).cuda()
        ut = torch.from_numpy(u).float().cuda()
        Dbt = torch.from_numpy(Db).float().cuda()

        #start_time = time.time()
        while np.max(np.absolute(v-vinf)) > 1/n:
            ut = torch.sparse.addmm(Dbt,Pt,ut)
            v = RW*v
            T = T + 1
        #print("--- %s seconds ---" % (time.time() - start_time))

        #Transfer to CPU and convert to numpy
        u = ut.cpu().numpy()

    else: #Use CPU

        #start_time = time.time()
        while np.max(np.absolute(v-vinf)) > 1/n:
            u = Db + P*u
            v = RW*v
            T = T + 1

            #Compute accuracy if all labels are provided
            if true_labels is not None:
                max_locations = np.argmax(u,axis=1)
                labels = (np.unique(g))[max_locations]
                labels[I] = g
                acc = accuracy(labels,true_labels,len(I))
                print('%d,Accuracy = %.2f'%(T,acc))
        
        #print("--- %s seconds ---" % (time.time() - start_time))

    return np.transpose(u),T



#Poisson L1 based on Split Bregman Method
#Does not work as well as PoissonMBO
def poissonL1(W,I,g,dataset,norm="none",lam=100,mu=1000,Nouter=30,Ninner=6,true_labels=None):

    n = W.shape[0]
    k = len(np.unique(g))

    #mu = mu*W.count_nonzero()/len(g)  #Normalize constants
    gamma = 1/lam

    WI,WJ,WV = sparse.find(W)
    B = sparse.coo_matrix((np.ones(len(WV),),(WI,WJ)),shape=(n,n)).tocsr() #Ones in all entries
    L = graph_laplacian(2*W.multiply(W),norm=norm)
    deg = degrees(W)
    dt = 1/np.max(deg)

    #Random initial labeling
    #u = random.rand(k,n)
    #u = ProjectToSimplex(u)
    u = np.zeros((k,n))

    #Set initial known labels
    J = np.zeros(n,)
    K = np.ones(n,)*g[0]
    J[I] = 1
    K[I] = g
    Kg,_ = LabelsToVec(K)
    Kg = Kg*J

    #Poisson parameters
    c = np.sum(Kg,axis=1)/len(I)
    b = np.transpose(Kg)
    b[I,:] = b[I,:]-c
    b = (mu/lam)*np.transpose(b)

    #Initialize u via Poisson learning
    u = np.zeros((k,n))
    L = graph_laplacian(W,norm='none')
    for j in range(k):
        u[j,:] = pcg_solve(L,b[j,:])
    u = np.transpose(np.transpose(u) - np.mean(u,axis=1))
   
    #Initialization
    V = list()
    R = list()
    gradu = list()
    for j in range(k):
        Gu = graph_gradient(W,u[j,:],I=WI,J=WJ,V=WV)
        gradu.append(Gu)
        V.append(Gu)
        R.append(Gu)

    #Main loop for Split Bregman iteration
    for i in range(Nouter):
        print('Outer:%d'%i)
        for s in range(Ninner):
            normV = 0*W
            for j in range(k):
                divVR = graph_divergence(R[j] - V[j],W)
                u[j,:] = pcg_solve(L,b[j,:] + divVR,x0=u[j,:],tol=1e-10)
                #for s in range(100):
                #    u[j,:] = u[j,:] + dt*(b[j,:] + divVR - u[j,:]*L)
                gradu[j] = graph_gradient(W,u[j,:],I=WI,J=WJ,V=WV)
                V[j] = gradu[j] + R[j]
                normV = normV + V[j].multiply(V[j]) 

            normV = normV.sqrt()

            #Shrinkage operation
            #normV^{-1} for nonzero entries (tricky to do in sparse format)
            #normV.eliminate_zeros(X)
            normVinv = normV - (1-1e-10)*B
            normVinv = -normVinv.log1p()
            normVinv = normVinv.expm1() + B
            
            C = normV.multiply(normVinv)
            #print(np.sum(C>0))
            #print(np.sum(C>0.9999))

            #Compute shrinkage factor
            #print(np.sum(normV>0))
            shrink = normV - gamma*B
            shrink = shrink.maximum(0)
            #print(np.sum(shrink>0))
            shrink = shrink.multiply(normVinv)
            
            #Apply shrinkage
            for j in range(k):
                V[j] = V[j].multiply(shrink)

        for j in range(k):
            R[j] = R[j] + gradu[j] - V[j]
         

        #Compute accuracy if all labels are provided
        if true_labels is not None:
            max_locations = np.argmax(u,axis=0)
            labels = (np.unique(g))[max_locations]
            labels[I] = g
            acc = accuracy(labels,true_labels,len(I))
            print('Accuracy = %.2f'%acc)

    return u

#Heap functions
#d = values in heap (indexed by graph vertex)
#h = heap (contains indices of graph elements in heap)
#p = pointers from graph back to heap (are updated with heap operations)
#s = number of elements in heap

#Sift up
#i = heap index of element to be sifted up
def SiftUp(d,h,s,p,i):

    pi = int(i/2)  #Parent index in heap
    while pi != 0:
        if d[h[pi]] > d[h[i]]:  #If parent larger, then swap
            #Swap in heap
            tmp = h[pi]
            h[pi] = h[i]
            h[i] = tmp

            #Update pointers to heap
            p[h[i]] = i     
            p[h[pi]] = pi

            #Update parent/child indices
            i = pi
            pi = int(i/2)
        else:
            pi = 0
            
            
#Sift down
#i = heap index of element to be sifted down
def SiftDown(d,h,s,p,i):

    ci = 2*i  #child index in heap
    while ci <= s:
        if d[h[ci+1]] < d[h[ci]] and ci+1 <= s:  #Choose smallest child
            ci = ci+1

        if d[h[ci]] < d[h[i]]:  #If child smaller, then swap
            #Swap in heap
            tmp = h[ci]
            h[ci] = h[i]
            h[i] = tmp

            #Update pointers to heap
            p[h[i]] = i     
            p[h[ci]] = ci

            #Update parent/child indices
            i = ci
            ci = 2*i
        else:
            ci = s+1

#Pop smallest off of heap
#Returns index of smallest and size of new heap
def PopHeap(d,h,s,p):
    
    #Index of smallest in heap
    i = h[1]

    #Put last element on top of heap
    h[1] = h[s]

    #Update pointer
    p[h[1]] = 1

    #Sift down the heap
    SiftDown(d,h,s-1,p,1)

    return i,s-1
     
#Push element onto heap
#i = Graph index to add to heap
def PushHeap(d,h,s,p,i):

    h[s+1] = i  #add to heap at end
    p[i] = s+1  #Update pointer to heap
    SiftUp(d,h,s+1,p,s+1)

    return s+1

#Nearest neighbor classifier (graph geodesic distance)
def nearestneighbor(W,I,g):

    
    k = len(I) #Number of labels
    n = W.shape[0] #Number of datapoints
    d = np.ones((n,))*1e10        #Distance function
    l = -np.ones((n,),dtype=int)    #Index of closest label

    #Reformat weight matrix W into form more useful for Dijkstra
    WI,WJ,WV = sparse.find(W)
    K = np.array((WJ[1:] - WJ[:-1]).nonzero()) + 1
    K = np.append(0,np.append(K,len(WJ)))

    try:  #Try to use fast C version of dijkstra, if compiled

        import cmodules.cgraphpy as cgp

        #Type casting and memory blocking
        d = np.ascontiguousarray(d,dtype=np.float64)
        l = np.ascontiguousarray(l,dtype=np.int32)
        WI = np.ascontiguousarray(WI,dtype=np.int32)
        WV = np.ascontiguousarray(WV,dtype=np.float64)
        K = np.ascontiguousarray(K,dtype=np.int32)
        I = np.ascontiguousarray(I,dtype=np.int32)

        cgp.dijkstra(d,l,WI,K,WV,I,1.0)

    except: #Use python version, which is slower

        #Initialization
        s = 0                           #Size of heap
        h = -np.ones((n+1,),dtype=int)  #Active points heap (indices of active points)
        A = np.zeros((n,),dtype=bool)   #Active flag
        p = -np.ones((n,),dtype=int)    #Pointer back to heap
        V = np.zeros((n,),dtype=bool)   #Finalized flag

        
        #Build active points heap and set distance = 0 for initial points
        for i in range(k):
            s = PushHeap(d,h,s,p,I[i])
            d[I[i]] = 0      #Initialize distance to zero
            A[I[i]] = True   #Set active flag to true
            l[I[i]] = I[i]   #Set index of closest label
        
        #Dijkstra's algorithm 
        while s > 0:
            i,s = PopHeap(d,h,s,p) #Pop smallest element off of heap

            #Finalize this point
            V[i] = True  #Mark as finalized
            A[i] = False #Set active flag to false

            #Update neighbors
            for j in WI[K[i]:K[i+1]]:
                if j != i and V[j] == False:
                    if A[j]:  #If j is already active
                        tmp_dist = d[i] + WV[j]
                        if tmp_dist < d[j]: #Need to update heap
                            d[j] = tmp_dist
                            SiftUp(d,h,s,p,p[j])
                            l[j] = l[i]
                    else: #If j is not active
                        #Add to heap and initialize distance, active flag, and label index
                        s = PushHeap(d,h,s,p,j)
                        d[j] = d[i] + WV[j]
                        A[j] = True  
                        l[j] = l[i]

    #Set labels based on nearest neighbor
    u = np.zeros((n,))
    u[I] = g
    u,_ = LabelsToVec(u[l])

    return u


#Computes accuracy of clustering
def clustering_accuracy(L,L_true):

    unique_classes = np.unique(L_true)
    num_classes = len(unique_classes)

    C = np.zeros((num_classes, num_classes), dtype=float)
    for i in range(num_classes):
        for j in range(num_classes):
            C[i][j] = np.sum((L == i) & (L_true != j))
    row_ind, col_ind = opt.linear_sum_assignment(C)

    return 100*(1-C[row_ind,col_ind].sum()/len(L))

#Spectral embedding
#Projects the graph to R^k via spectral projection 
#Method can be 'unnormalized', 'ShiMalik', or 'NgJordanWeiss'
def spectral_embedding(W,k,method='NgJordanWeiss'):

    n = W.shape[0]

    if method == 'unnormalized':
        L = graph_laplacian(W,norm='none') 
        vals, vec = sparse.linalg.eigs(L,k=k,which='SM')
        vec = vec.real
        vals = vals.real
    elif method == 'ShiMalik':
        D = degree_matrix(W)
        L = graph_laplacian(W,norm='none') 
        vals, vec = sparse.linalg.eigs(L,M=D,k=k,which='SM')
        vec = vec.real
        vals = vals.real
    elif method == 'NgJordanWeiss':
        L = graph_laplacian(W,norm='normalized') 
        vals, vec = sparse.linalg.eigs(L,k=k,which='SM')
        vec = vec.real
        vals = vals.real
        norms = np.sum(vec*vec,axis=1)
        T = sparse.spdiags(norms**(-1/2),0,n,n)
        vec = T@vec  #Normalize rows

    return vec

#Spectral Clustering
def spectral_cluster(W,k,method='NgJordanWeiss',extra_dim=0):

    V = spectral_embedding(W,k+extra_dim,method=method)
    kmeans = cluster.KMeans(n_clusters=k).fit(V)
    #V = spectral_embedding(W,k,method=method)
    #kmeans = cluster.KMeans(n_clusters=k).fit(V)
    return kmeans.labels_


#INCRES clustering
#Bresson, Xavier, et al. "An incremental reseeding strategy for clustering." International Conference on Imaging, Vision and Learning based on Optimization and PDEs. Springer, Cham, 2016.
#W = weight matrix 
def incres_cluster(W,k,speed,T,labels):

    n = W.shape[0]

    #Increment
    Dm = np.maximum(int(speed*1e-4*n/k),1)
    
    #Random initial labeling
    u = random.randint(0,k,size=n)

    #Initialization
    F = np.zeros((n,k))
    J = np.arange(n).astype(int)

    #Random walk transition
    D = degree_matrix(W,p=-1)
    P = W*D

    m = int(1)
    for i in range(T):
        #Plant
        F.fill(0)
        for r in range(k):
            I = u == r
            ind = J[I]
            F[ind[random.choice(np.sum(I),m)],r] = 1
        
        #Grow
        while np.min(F) == 0:
            F = P*F

        #Harvest
        u = np.argmax(F,axis=1)

        #Increment
        m = m + Dm
            
        #Compute accuracy
        if labels is not None: 
            acc = clustering_accuracy(u,labels)
            print("Iteration "+str(i)+": Accuracy = %.2f" % acc+"%%, #seeds= %d" % m)

    return u

#Graph-based clustering
#W = sparse weight matrix describing graph
#method = SSL method
#   Options: incres
def graph_clustering(W,k,true_labels=None,method="incres",speed=5,T=100,extra_dim=0):

    n = W.shape[0]
    
    #Symmetrize W, if not already symmetric
    W = (W + W.transpose())/2

    #Check if connected
    num_comp,comp = csgraph.connected_components(W)
    if num_comp != 1:
        print('Warning: Graph is not connected!')

    #Clustering
    if method=="INCRES":
        labels = incres_cluster(W,k,speed,T,true_labels)
    elif method=="Spectral":
        labels = spectral_cluster(W,k,method="unnormalized",extra_dim=extra_dim)
    elif method=="SpectralShiMalik":
        labels = spectral_cluster(W,k,method="ShiMalik",extra_dim=extra_dim)
    elif method=="SpectralNgJordanWeiss":
        labels = spectral_cluster(W,k,method="NgJordanWeiss",extra_dim=extra_dim)
    else:
        print("Invalid choice of clustering method.")
        sys.exit()

    return labels


#Graph-based semi-supervised learning
#W = sparse weight matrix describing graph
#I = indices of labeled datapoints
#g = values of labels
#method = SSL method
#   Options: laplace, poisson, poisson_nodeg, wnll, properlyweighted, plaplace, randomwalk
def graph_ssl(W,I,g,D=None,beta=None,method="laplace",p=3,alpha=2,zeta=1e7,r=0.1,epsilon=0.05,X=None,plaplace_solver="GradientDescentCcode",norm="none",true_labels=None,eigvals=None,eigvecs=None,dataset=None,T=0,use_cuda=False,return_vector=False):

    one_shot_methods = ["mbo","poisson","poissonmbo","poissonl1","nearestneighbor","poissonmbobalanced","volumembo","poissonvolumembo","dynamiclabelpropagation","sparselabelpropagation","centeredkernel"]

    n = W.shape[0]

    method = method.lower()

    if beta is None:
        beta = np.ones((len(np.unique(g)),))

    #Symmetrize W, if not already symmetric
    W = (W + W.transpose())/2

    #Check if connected
    num_comp,comp = csgraph.connected_components(W)
    if num_comp != 1:
        print('Warning: Graph is not connected!')
    
    #One shot methods
    if method in one_shot_methods:

        if method=="mbo":
            u = multiclassMBO(W,I,g,eigvals,eigvecs,dataset,true_labels=true_labels)
        elif method=="volumembo":
            u = volumeMBO(W,I,g,dataset,beta,T)
        elif method=="poissonvolumembo":
            u = poisson_volumeMBO(W,I,g,dataset,beta,T)
        elif method=="poissonmbo":
            u = poissonMBO(W,I,g,dataset,beta*0,true_labels=true_labels,temp=T,use_cuda=use_cuda)
        elif method=="poissonmbobalanced":
            u = poissonMBO(W,I,g,dataset,beta,true_labels=true_labels,temp=T,use_cuda=use_cuda)
        elif method=="poissonl1":
            u = poissonL1(W,I,g,dataset,true_labels=true_labels)
        elif method=="poisson":
            u,_ = poisson(W,I,g,true_labels=true_labels,use_cuda=use_cuda)
        elif method=="dynamiclabelpropagation":
            u = DynamicLabelPropagation(W,I,g,true_labels=true_labels)
        elif method=="sparselabelpropagation":
            u = SparseLabelPropagation(W,I,g,true_labels=true_labels)
        elif method=="centeredkernel":
            u = CenteredKernel(W,I,g,true_labels=true_labels)
        elif method=="nearestneighbor":
            #USe distance matrix if provided, instead of weight matrix
            if D is None:
                u = nearestneighbor(W,I,g)
            else:
                u = nearestneighbor(D,I,g)

    else:  #One vs rest methods

        k = len(np.unique(g))  #Number of labels
        u = np.zeros((k,n))
        i = 0
        for l in np.unique(g):
            #Convert to +1/-1 labels
            h = 2*(g==l)-1 

            #Solve binary classification problem
            if method=="laplace":
                v = laplace_solve(W,I,h,norm=norm)
            elif method=="shift":
                v = shift_solve(W,I,h)       
            elif method=="meanshift":
                v = meanshift_solve(W,I,h)       
            elif method=="wnll":
                v = wnll_solve(W,I,h)
            elif method=="properlyweighted":
                if X is None:
                    print("Must supply raw data points for properly weighted Laplacian.")
                    sys.exit()
                v = properlyweighted_solve(W,I,h,X,alpha,zeta,r)
            elif method=="plaplace":
                v = plaplace_solve(W,I,h,p,sol_method=plaplace_solver,norm=norm)
            elif method=="randomwalk":
                v = randomwalk_solve(W,I,h,epsilon)
            else:
                print("Invalid choice of SSL method.")
                sys.exit()

            #Update labels
            u[i,:] = v
            i = i+1

    if return_vector:
        labels = np.transpose(u)
    else:
        #Select labels
        max_locations = np.argmax(u,axis=0)
        labels = (np.unique(g))[max_locations]

        #Make sure to set labels at labeled points
        labels[I] = g


    return labels
    confidence = usort[0,:] - usort[1,:]

#Read numerical data from csv file
def csvread(filename):
    
    X = [] 
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        n = 0
        for row in csv_reader:
            if not row[0]=='Date/Time':
                X += [float(i) for i in row]
                m = len(row)
                n += 1

    return np.array(X).reshape((n,m))

#Compute average and standard deviation of accuracy over many trials
#Reads data from csv file filename
#Returns accuracy (acc), standard deviation (stddev) and number of labeled points (N)
def accuracy_statistics(filename):

    X = csvread(filename)
    N = np.unique(X[:,0])

    acc = []
    stddev = []
    quant = []
    for n in N:
        Y = X[X[:,0]==n,1]
        Y = np.sort(Y)
        acc += [np.mean(Y)]
        quant += [Y[int(3*len(Y)/4)]]
        stddev += [np.std(Y)]
        #print("%.1f (%.1f)"%(np.mean(Y),np.std(Y)), end="&")

    num_trials = len(X[:,0])/len(N) 
    return acc,stddev,N,quant,num_trials

#Makes an accuracy table to be included in LaTeX documents
#dataset = name of dataset
#ssl_methods = list of names of methods to compare
def accuracy_table_icml(dataset,ssl_method_list,legend_list,num_of_classes,testerror=False,savefile='tables.tex',title='',quantile=False,append=False):

    #Retrieve number of different label rates m
    accfile = "Results/"+dataset+"_"+ssl_method_list[0]+"_accuracy.csv"
    acc,stddev,N,quant,num_trials = accuracy_statistics(accfile)
    m = len(N)

    #Determine best algorithm at each label rate
    best = [None]*m
    best_score = [0]*m
    i=0
    for ssl_method in ssl_method_list:
        accfile = "Results/"+dataset+"_"+ssl_method+"_accuracy.csv"
        acc,stddev,N,quant,num_trials = accuracy_statistics(accfile)
        if quantile:
            acc = quant
        for j in range(m):
            if acc[j] > best_score[j]:
                best_score[j] = acc[j]
                best[j] = i
        i+=1
    
    if append:
        f = open(savefile,"r")
        lines = f.readlines()
        f.close()
        f = open(savefile,"w")
        f.writelines([item for item in lines[:-1]])
    else:
        f = open(savefile,"w")
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage[T1]{fontenc}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage[margin=1in]{geometry}\n")
        f.write("\\begin{document}\n")

    f.write("\n\n\n")
    f.write("\\begin{table*}[t]\n")
    f.write("\\caption{"+title+": Average accuracy scores over %d trials with standard deviation in brackets.}\n"%num_trials)
    f.write("\\label{tab:"+title+"}\n")
    f.write("\\vskip 0.15in\n")
    f.write("\\begin{center}\n")
    f.write("\\begin{small}\n")
    f.write("\\begin{sc}\n")
    f.write("\\begin{tabular}{l")
    for i in range(m):
        f.write("l")
    f.write("}\n")
    f.write("\\toprule\n")
    f.write("\\# Labels per class")
    for i in range(m):
        f.write("&\\textbf{%d}"%int(N[i]/num_of_classes))
    f.write("\\\\\n")
    f.write("\\midrule\n")
    i = 0
    for ssl_method in ssl_method_list:
        f.write(legend_list[i].ljust(15))
        accfile = "Results/"+dataset+"_"+ssl_method+"_accuracy.csv"
        acc,stddev,N,quant,num_trials = accuracy_statistics(accfile)
        for j in range(m):
            if best[j] == i: 
                f.write("&{\\bf %.1f"%acc[j]+" (%.1f)}"%stddev[j])
            else:
                f.write("&%.1f"%acc[j]+" (%.1f)      "%stddev[j])
        f.write("\\\\\n")
        i+=1

    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{sc}\n")
    f.write("\\end{small}\n")
    f.write("\\end{center}\n")
    f.write("\\vskip -0.1in\n")
    f.write("\\end{table*}")
    f.write("\n\n\n")
    f.write("\\end{document}\n")
    f.close()

def plot_graph(X,W,l=None):
#Other colormaps, coolwarm, winter, Set3, tab20b, rainbow

    #plt.ion()
    colors = np.array([[1.0,0,0],[0,0.9,0]])
    plt.rcParams['figure.facecolor'] = 'navy'

    n = W.shape[0]
    I,J,V = sparse.find(W)

    for i in range(len(I)):
        xval = [X[I[i],0],X[J[i],0]]
        yval = [X[I[i],1],X[J[i],1]]
        #plt.plot(xval,yval, color='black', linewidth=0.15, markersize=0)
        plt.plot(xval,yval, color=[0.5,0.5,0.5], linewidth=0.5, markersize=0)

    if l is None:
        #plt.scatter(X[:,0],X[:,1], s=30, cmap='Paired')
        plt.scatter(X[:,0],X[:,1], s=8, zorder=3)
    else:
        #plt.scatter(X[:,0],X[:,1], s=30, c=l, cmap='Paired')
        plt.scatter(X[:,0],X[:,1], s=8, c=colors[l,:], zorder=3)

    plt.axis("off")

#plot average and standard deviation of accuracy over many trials
#dataset = name of dataset
#ssl_methods = list of names of methods to compare
def accuracy_plot(dataset,ssl_method_list,legend_list,num_of_classes,title=None,errorbars=False,testerror=False,savefile=None,loglog=False):

    #plt.ion()
    plt.figure()
    if errorbars:
        matplotlib.rcParams.update({'errorbar.capsize': 5})
    matplotlib.rcParams.update({'font.size': 16})
    styles = ['^b-','or-','dg-','sk-','pm-','xc-','*y-']
    i = 0
    for ssl_method in ssl_method_list:
        accfile = "Results/"+dataset+"_"+ssl_method+"_accuracy.csv"
        acc,stddev,N,quant,num_trials = accuracy_statistics(accfile)
        if testerror:
            acc = 100-acc
            #z = np.polyfit(np.log(N),np.log(acc),1)
            #print(z[0])
        if errorbars:
            plt.errorbar(N/num_of_classes,acc,fmt=styles[i],yerr=stddev,label=legend_list[i])
        else:
            if loglog:
                plt.loglog(N/num_of_classes,acc,styles[i],label=legend_list[i])
            else:
                plt.plot(N/num_of_classes,acc,styles[i],label=legend_list[i])
        i+=1
    plt.xlabel('Number of labels per class')
    if testerror:
        plt.ylabel('Test error (%)')
        plt.legend(loc='upper right')
    else:
        plt.ylabel('Accuracy (%)')
        plt.legend(loc='lower right')
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.grid(True)
    if savefile is not None:
        plt.savefig(savefile)
    else:
        plt.show()


#Select labels based on a ranking
#Prodces a label permutation with 1 trial with same variations of #labels per class as the label permutation perm provided as input
def SelectLabels(labels,permold,rank):

    perm = permold

    #Number of classes
    L = np.unique(labels)
    k = len(L)
    n = len(labels)

    m = len(permold)
    num = np.zeros((m,))
    for i in range(m):
        num[i] = len(permold[i])

    
    num,unique_perm = np.unique(num,return_index=True)

    perm = list()
    for i in unique_perm:
        p = permold[i]
        pl = labels[p]
        ind = []
        for l in L:
            numl = np.sum(pl == l)
            K = labels == l
            c = np.argsort(-rank[K])
            j = np.arange(0,n)[K]
            ind = ind + j[c[:numl]].tolist()
        ind = np.array(ind)
        perm.append(ind)


    return perm

#PageRank algorithm
def PageRank(W,alpha):

    n = W.shape[0]

    u = np.ones((n,))
    v = np.ones((n,))

    D = degree_matrix(W,p=-1)
    P = np.transpose(D*W)

    err = 1
    while err > 1e-10:
        w = alpha*P*u + (1-alpha)*v
        err = np.max(np.absolute(w-u))
        u = w


    return u









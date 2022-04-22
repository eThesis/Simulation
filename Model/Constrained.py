from packages import *


# Model takes 5 inputs
# epsilon used in convergence criterion `converged'
# run_EM returns the model density

class ConstrainedModel(Data):
    def __init__(self, raw_Data_Matrix=[[1,2, 3],[3,4, 5]], cluster_Count=3, latent_Count=5, epsilon=1, rowConstraints = 3):
        super().__init__(raw_Data_Matrix)
        self.epsilon = epsilon
        self.G = cluster_Count
        self.q = latent_Count
        self.Z = init_Z(self.sample_Size, self.G)   # component membership: each row has exactly a single 1 initially
        self.MixProp = init_MixProp(self.G)     # this is a vector 
        self.Lamb = np.random.rand(self.dim, self.q)
        self.Psi =  np.identity(self.dim) #init_Psi(self.dim) #np.identity(self.dim) #init_Psi(self.dim)
        self.Mean = 0.00001 * init_Mean(self.q, self.G)   # mean stored in column 
        self.Lg = init_Cholesky(self.G, self.q)[0]
        self.D = np.identity(self.q)    
        #self.Lg = 10 * init_Lg(self.G, self.q) 
        #self.Dg = init_Dg(self.G, self.q)
        self.L1, self.L2 = 0, self.log_likelihood()    # stores sucessive logLike (see also > def converged)
        self.null = rowConstraints

    def gaussian_density(self, X, dim, mean, covariance):
        c = (2 * const.pi) ** dim
        print("Determinant", abs(det(covariance)))
        k = 1 / np.sqrt(c * abs(det(covariance)))
        diff = (X - mean).reshape(dim, 1)
        diffT = (X - mean).reshape(1, dim)  # transpose
        expon = - diffT @ inv(covariance) @ diff / 2
        print("\n\n\n\nGaussian Density ", (k * np.exp(expon))[0][0])
        return (k * np.exp(expon))[0][0]

    def component_density(self, g, X):
        """g-th component (indexed from 0) density in mixture model, X is a vector"""
        meanVect = self.Lamb @ self.Mean[:, g]
        invL = inv(self.Lg[g])
        D = self.D
        #covMat = 100 * sklearn.datasets.make_spd_matrix(self.dim)
        #self.Psi = covMat - self.Lamb @ self.Lamb.T
        #self.ind = 1
        covMat =  self.Lamb @ invL @ D @ invL.T @ self.Lamb.T + self.Psi 
        return self.gaussian_density(X, self.dim,  meanVect, covMat)  #multivariate_normal.pdf(X, meanVect, 100 * covMat, allow_singular=True) #self.gaussian_density(X, self.dim, meanVect, covMat)  #multivariate_normal.pdf(X, meanVect, covMat, allow_singular=True) #

    def mixture_density(self, X):   
        density = 0
        for cluster in range(self.G):
            density += self.component_density(cluster, X)   
            print("\n\n\n\n\nDensity is ", density)
        return density 
    
    def log_likelihood(self):   # assume data not labeled 
        likelihood = 1
        for dataIndex in range(self.sample_Size):
            likelihood *= self.mixture_density(self.data_Matrix[dataIndex, :])
            print("\n\n\n\n\n\nLikeli is ", likelihood)
        return np.log(likelihood)

    def converged(self):
        delta = self.L2 - self.L1
        return abs(delta) < self.epsilon # true or false, did EM converge?

    # computes n_g, Beta_g and S_g to be used in up_Mean
    def nBetaS(self, g):    
        """g counts from 1"""
        while type(g) != int or not (0 <= g < self.G):
            g = input('Please enter a cluster index, between 0 and ' + str(self.G - 1) + ' inclusive: ')
            try:
                g = int(g) 
            except ValueError:
                continue
        # useful parameters
        L, D, Lamb = self.Lg[g], self.D, self.Lamb
        Omega = inv(L) @ D @ inv(L.T)   # dim q x q
        A = Lamb @ Omega @ Lamb.T + self.Psi    # dim p x p
        # computation
        Beta = Omega @ Lamb.T @ inv(A)  # dim q x p
        n_g = np.dot(self.Z[:, g], np.ones(self.sample_Size))    # number
        S = np.random.rand(self.q, self.q) # dim q x q
        if n_g != 0:
            S *= 0
            for i in range(self.sample_Size):   # i = Ind(sample i)
                J = self.data_Matrix[i,:] - Lamb @ self.Mean[:,g]       # dim p vevtor 
                J1 = J.reshape((self.dim, 1))
                J2 = J.reshape((1, self.dim))
                Q = np.identity(self.q) - Beta @ Lamb   # dim q x q
                K = self.Z[i, g]    # number 
                S += K * (Q @ Omega + Beta @ J1 @ J2 @ Beta.T)   # dim q x q
            return n_g, Beta, S / n_g
        return n_g, Beta, S

    def up_Mean(self):  
        """update mean for each cluster"""
        for g in range(self.G):
            n, b = self.nBetaS(g)[:-1] 
            mu = self.Lamb @ self.Mean[:, g]
            diff = 0
            for i in range(self.sample_Size):
                X_i = self.data_Matrix[i,:]
                diff += self.Z[i,g] * b @ (X_i - mu)
            if n > 0:   # catches division by zero 
                self.Mean[:, g] += diff / n     # recall mean is stored as column vector of self.Mean matrix
        return 0    # update complete with normal termination

    def soft_Zi(self, i):
        for g in range(self.G):
            w = self.MixProp[g] # weight
            x = self.data_Matrix[i,:]
            self.Z[i, g] = w * self.component_density(g, x) / self.mixture_density(x)
        return 0
    
    def up_Z(self):
        """update soft component membership"""
        for data in range(self.sample_Size):
            self.soft_Zi(data)
        #print(self.Z) # for testing purposes
        return 0
    
    # unconstrained
    def up_Row_Constrained(self, g, r):    # r = Ind(row) and g index starts from 0
        """updates a row"""
        while type(g) != int or not 0 <= g < self. G:
            g  = input('Enter component index between 0 and ' + str(self.G-1) + ' ')
            try:
                g = int(g)
            except ValueError:
                continue
        while type(r) != int or not 0 < r < self.q:     # row 1 has nothing to solve
            print(r)
            r  = input('Enter row index between 0 and ' + str(self.q - 1) + ' ')
            try:
                r = int(r)
            except ValueError:
                continue
        Sg = self.nBetaS(g)[2]
        if r <= self.null: # constrain parameter
            # solves Ax = b
            A = Sg[:r,:r]   # dim = r x r
            b = Sg[r, :r]   # row r up to first r entries, counts from 0 -> dim = 1 x r
            x = - inv(A.T) @ b
        else:
            A = Sg[self.null:r, :self.null]
            b = Sg[self.null :self.r]
            x = - inv(A.T) @ b
        # update Lg[g] row r
        for column in range(r):
            self.Lg[g][r,column] = x[column]
        return 0    # normal termination
    
    def up_L_Constrained(self):
        """for each component matrix, and update each row"""
        for g in range(self.G):
            for r in range(1, self.q):   # row 0 requires no update
                self.up_Row_Constrained(g,r)
        return 0

    def get_ng(self, g):
        ng = 0
        for i in range(self.sample_Size):
            ng += self.Z[i, g]
        return ng

    def up_MixProp(self):
        for g in range(self.G):
            self.MixProp[g] = self.get_ng(g) / self.sample_Size
        return 0

    def up_D(self):
        D = np.copy(self.D)
        for g in range(self.G):
            ng, b, S = self.nBetaS(g)
            L = self.Lg[g]
            P = np.identity(self.q) - b @ self.Lamb   # dim q x q
            FirstMatrix = L @ P @ L.T @ self.D
            SecondMatrix = np.zeros((self.q, self.q))
            for i in range(self.sample_Size):
                z = self.Z[i, g]
                Q = L @ b @ (self.data_Matrix[i,:] - self.Lamb @ self.Mean[:,g])        # potential dimension mismatch
                SecondMatrix += z * Q @ Q.T
            D = FirstMatrix + SecondMatrix / ng    
            D += diag(diag(D))  # a diagonal matrix
        self.D = D / self.sample_Size
        return 0
    
    def up_Lamb(self):
        A = np.zeros((self.dim, self.q))
        B = np.zeros((self.q, self.q))
        for g in range(self.G):
            beta = self.nBetaS(g)[1] # have it ready once for all i
            Q = np.identity(self.q) - beta @ self.Lamb
            invL = inv(self.Lg[g])
            R = invL @ self.D @ invL.T  # simplify by computing inverse once 
            for i in range(self.sample_Size):
                z = self.Z[i, g]
                X = self.data_Matrix[i,:]
                P = self.Mean[:, g] + beta @ (X - self.Lamb @ self.Mean[:, g])        # potential dim mismatch
                A += z * X.reshape((self.dim, 1)) @ P.reshape((1, self.q))
                P2 = P.reshape((self.q, 1))
                B += z * (Q @ R + P2 @ P2.T) 
        self.Lamb = A @ inv(B)
        return 0

    def up_Psi(self):
        # used = {i, g, l, b, A, B, C, E, X, z, Psi, K}
        Psi = np.zeros((self.dim, self.dim))
        l = self.Lamb   # lambda
        for g in range(self.G):
            b = self.nBetaS(g)[1]   # beta    
            A = np.identity(self.q) - b @ l  
            invL = inv(self.Lg[g])
            B = invL @ self.D @ invL.T     
            for i in range(self.sample_Size):
                z = self.Z[i, g]
                X = self.data_Matrix[i, :]
                C = self.Mean[:, g] + b @ (X - l @ self.Mean[:, g])
                K = l @ C @ X
                E = A @ B + C @ C.T
                # finally compute Psi
                X = np.copy(X)
                X2 = X.reshape(self.dim, 1)
                Psi += z * (X2 @ X2.T - K - K.T + l @ E @ l.T)
        self.Psi = diag(diag(Psi))
        return 0
    
    def up_log(self):
        L1 = self.L2
        self.L1, self.L2 = L1, self.log_likelihood()
    
                
    def update(self):
        self.up_Mean()
        self.up_L_Constrained()
        self.up_Lamb()
        self.up_D()
        self.up_Psi()
        self.up_log()
        self.up_MixProp()
        self.up_Z()
        return 0

    def run_EM(self):
        while (not self.converged()):
            print('Round', i)
            self.update() 
        print('EM completed, model has been fitted.')
        return self.mixture_density   # returns density (a function)
# testing
def test():
    pass

if __name__ == '__main__':
    test()
    

    
    


from new_kernels import Kernel

import numpy as np

from scipy import optimize
from scipy.linalg import solve
from scipy.spatial.distance import pdist, squareform, euclidean
from matplotlib import pyplot as plt
import pdb

import GPy

class GP:

    def __init__(self, time_points, values, kernels, noise_level = 1.0):
        
        self.time_points = time_points
        self.values = values
        self.noise_level = noise_level
        
        self.kernels = []
        self.kernels += kernels
        
        self.cov = self.get_cov(self.kernels)
        self.margin_likelihood = self.get_margin_likelihood()

    def get_cov(self, kernels):
        X = self.time_points
        N = len(X)
        K = np.zeros((N, N)) * 1.0

        for i in range(len(kernels)):

            K += self.cal_K(kernels[i])

        return K

    def get_margin_likelihood(self):
        if self.cov is None:
            self.cov = self.get_cov()
        X = self.time_points
        Y = self.values
        N = len(X)
        K = self.cov
        noise_level = 1.0
        C = K + np.power(noise_level, 2) * np.identity(N)
        L = np.log(np.linalg.det(C)) * 0.5 + 0.5 * np.dot(np.dot(Y.T, np.linalg.inv(C)), Y) + N * 1.0 / 2 * np.log(2 * np.pi)
        return L

    def cal_K(self, kernel):
        X = self.time_points

        if kernel.kernel_type == 'SE':
            pars = kernel.pars
            R = (X.T - X) / pars[0]
            R = np.power(R, 2)
            K = np.power(pars[1], 2) * np.exp(-R)
            
        return K


    def optimization_function(self, parameters):
        

        X = self.time_points
        Y = self.values
        
        N = len(X)
        K = np.zeros((N, N)) * 1.0
        
        noise_level = parameters[-1]

        new_kernels = []
        j = 0

        for i in range(len(self.kernels)):

            kernel_type = self.kernels[i].kernel_type
            if kernel_type == "SE":
                new_kernel = Kernel(kernel_type, parameters[j], parameters[j+1])
                j = j + 1
            
            new_kernels.append (new_kernel)


        K = self.get_cov(new_kernels)
        C = K + np.power(noise_level, 2) * np.identity(N)
        L = np.log(np.linalg.det(C)) * 0.5 + 0.5 * np.dot(np.dot(Y.T, np.linalg.inv(C)), Y) + N * 1.0 / 2 * np.log(2 * np.pi)
        return L

    def optimize(self):
        """
        Optimize marginal likelihood 
        Update parameters for kernels
        """
        pars = []
        for ker in self.kernels:
            pars += ker.pars
            
        pars.append(self.noise_level)

        pars = optimize.fmin_bfgs(self.optimization_function, pars)

        print pars


if __name__ == '__main__':
    
    # results from GPy 
    X = np.array([1,2,3,4]).reshape(-1,1)
    Y = np.sin(X) + np.random.randn(4,1)

    # results from GPy
    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1)
    m = GPy.models.GPRegression(X,Y,kernel)
    m.optimize()
    print "Correct parameters:"
    print m

    # results from GP 
    k1 = Kernel("SE", 1,1)

    gp1 = GP(time_points = X, values = Y, kernels =[k1])

    gp1.optimize()


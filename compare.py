# compare the optimization results from GP, new_GP, and GPy
from new_gp import GP
from new_kernels import Kernel
import GPy

import numpy as np 
import scipy 

from matplotlib import pyplot as plt
from scipy.linalg import solve
from scipy.spatial.distance import pdist, squareform,euclidean
from matplotlib import pyplot as plt

# results from GPy 
X = np.array([1,2,3,4]).reshape(-1,1)
Y = np.sin(X) + np.random.randn(4,1)
kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1)
m = GPy.models.GPRegression(X,Y,kernel)

m.optimize()
print "Correct parameters:"
print m

# results from GP 
k1 = Kernel("SE", 1,1)
gp1 = GP(time_points = X, values = Y, kernels =[k1])

gp1.optimize()


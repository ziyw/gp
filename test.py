from scipy import optimize
import numpy as np 
# from gp import GP
import GPy

import numpy as np 
import scipy 
from kernels import Kernel
from matplotlib import pyplot as plt
from scipy.linalg import solve
from scipy.spatial.distance import pdist, squareform,euclidean
from matplotlib import pyplot as plt

X = np.array([1,2,3,4]).reshape(-1,1)
Y = np.sin(X) # np.random.randn(20,1)*0.05

def get_cov(x):
	h = x[0]
	l = x[1]

	P = X * 1.

	R = (P.T - P)/l
	R = np.power(R, 2)
	K = np.power(h,2) * np.exp(-R)
	return K

def se_function(x):

	variance = x[0]
	length = x[1]
	noise_level = x[2]

	K = get_cov(x)
	(N,_)= K.shape
	C = K + noise_level * noise_level * np.identity(N)

	L = np.log(np.linalg.det(C))*0.5 + 0.5 * np.dot(np.dot(Y.T, np.linalg.inv(C)), Y) +  N*1. / 2 * np.log(2* np.pi)

	return L

# print optimize.fmin_bfgs(se_function, [1,1,1])

X = np.matrix([1,2,3,4])
print X - X.T

# print optimize.fmin_bfgs(g,[2,2])
# print optimize.fmin_bfgs(f, [3, 4], fprime=fprime)
# X = np.array([1,2,3,4]).reshape(-1,1)
# # X = np.random.uniform(-3.,3.,(20,1))
# Y = np.sin(X) # np.random.randn(20,1)*0.05


# # # GPy test plot
# # kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
# # m = GPy.models.GPRegression(X,Y,kernel)
# # m.optimize()
# # m.plot()

# # plt.show(block=True)

# # gp plot 
# X = X.reshape(4,)
# Y = Y.reshape(4,)
# k1 = Kernel("SE",np.sqrt(2),1)
# gp1 = GP(time_points = X.T, values = Y.T, kernel = k1)
# # gp1.plot()
# gp1.plot()


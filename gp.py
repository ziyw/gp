# Gaussian process regression 
# GPR 
import numpy as np 
import scipy 
import kernels
from matplotlib import pyplot as plt

from scipy.linalg import solve
from scipy.spatial.distance import pdist, squareform,euclidean

class GP:

	def __init__(self,X,y,sigma = 1):
			
		self.X = X 
		self.y = y
		self.noise_level = sigma

		# N number of training points
		self.N = X.size
		
	def regression(self, x_test, kernel_type = "SE", *arguments):
		# Gaussian process regression 
		x_test.astype("float")
		X_all = np.append(self.X, x_test).reshape(-1,1)
		self.kernel = kernels.Kernel(X_all, X.size + x_test.size)
		
		if kernel_type == "SE":

			arg = list(arguments)
			h,l = arg[0],arg[1]
			# K matrix contains K(x,x), K (x*,x) and K(x*,x*)
			K_all = self.kernel.SE(h,l)
		
		if kernel_type ==  "RQ":
			arg = list(arguments)
			h,alpha,l = arg[0],arg[1],arg[2]
			K_all = self.kernel.RQ(h,alpha,l)

		if kernel_type == "per_SE":
			arg = list(arguments)
			h,w,l = arg[0],arg[1],arg[2]
			K_all = self.kernel.per_SE(h,w,l)

		N = self.N

		# number of test points
		tn = x_test.size

		K = K_all[:N,:N]
		cov_k_test_K = K_all[N:N+tn,0:N]
		cov_k_test = K_all[N:N+tn,N:N+tn].diagonal()
		cov_K = K_all[:N,:N]
		# calculate A 	

		# need to add the noise level later
		s = self.noise_level 
		A = cov_K + np.identity(N)* s * s
		
		# calculate L 
		L = np.linalg.cholesky(A)
		y = self.y

		alpha = solve(L.T,solve(L, y)).reshape(-1,1)
		
		# calculate f*
		print cov_k_test_K.shape

		mean = np.matmul(cov_k_test_K, alpha)
		v = solve(L, cov_k_test_K.T)

		print cov_k_test

		var = cov_k_test - np.matmul(v.T, v)

		# marginal likelihood 
		p = - 1.0 / 2.0 * np.dot(y , alpha) - np.sum(np.log(L.diagonal())) - (N/2.0 *  np.log (2 * np.pi))

		return mean, var, p

	def classfication():
		pass 

if __name__ == '__main__':

	# input data for GP 
	
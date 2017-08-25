# Gaussian process regression 
# GPR 
import numpy as np 
import scipy 

from scipy.linalg import solve
from scipy.spatial.distance import pdist, squareform,euclidean

class GP:
	def __init__(self, X, y,x_test, sigma_n, kernal):
		self.X = X
		self.y = y
		self.x_test = x_test 
		self.sigma_n = sigma_n
		self.kernal = kernal

	def regression(self):
		# Gaussian process regression 

		X = self.X
		y = self.y
		x_test = self.x_test
		sigma = self.sigma_n
		kernal = self.kernal

		N,D = X.shape 

		# calculate K and k*
		if kernal == "SE":

			# To calculate the euclidean distance faster, concatenate two matrix together 
			# cov(k*, K ), cov(K,K) and cov(k*, k*)

			X_all = np.concatenate((X, x_test), axis=0)

			pairwise_dists = squareform(pdist(X_all, 'euclidean'))
			pairwise_dists = pairwise_dists ** 2 

			# Kernel for X_all, 
			# K is [1:N, 1:N] 
			# k* is the last row K[:-1,:-1]

			K = np.exp(- pairwise_dists / 2)
			cov_k_test_K = K[-1,:-1]
			cov_K = K[:-1,:-1]
			cov_k_test = K[-1,-1]


		# calculate A 	
		A = cov_K + np.identity(D) * sigma * sigma 
		# calculate L 
		L = np.linalg.cholesky(A)

		alpha = solve(L.T,solve(L, y))
		
		# calculate f*
		mean = cov_k_test_K.T * alpha
		v = solve(L, cov_k_test_K)

		var = cov_k_test - v.T * v

		# marginal likelihood 
		p = - 1.0 / 2.0 * np.dot(y , alpha) - np.sum(np.log(L.diagonal())) - (N/2.0 *  np.log (2 * np.pi))

		return mean, var, p


if __name__ == '__main__':
	X = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
	y = np.array([1,1,1])
	sigma = 1 
	x_test = np.matrix([17,18,19])

	gp = GP(X, y,x_test,sigma,"SE")
	print gp.regression()


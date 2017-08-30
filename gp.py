# Gaussian process regression 
# GPR 
import numpy as np 
import scipy 
from kernels import Kernel
from matplotlib import pyplot as plt
from scipy.linalg import solve
from scipy.spatial.distance import pdist, squareform,euclidean


class GP:


	def GPR(self, time_points, values, predict_point, kernel, noise_level = 1):
		'''
		One point at a time 
		'''
		X = np.append(time_points, predict_point).reshape(-1,1)
		N = time_points.size

		if kernel.type == 'SE':
			K = kernel.cal_SE(X)

		cov_K = K[:N,:N]
		cov_k_K = K[:N,-1]
		cov_k = K[-1,-1]

		# need to add the noise level later
		s = noise_level 
		A = cov_K + np.identity(N)* s * s
		
		# calculate L 
		L = np.linalg.cholesky(A)
		y = values

		alpha = solve(L.T,solve(L, y)).reshape(-1,1)

		mean = np.matmul(cov_k_K, alpha)
		v = solve(L, cov_k_K.T)

		var = cov_k - np.matmul(v.T, v)

		# marginal likelihood 
		p = - 1.0 / 2.0 * np.dot(y , alpha) - np.sum(np.log(L.diagonal())) - (N/2.0 *  np.log (2 * np.pi))

		return mean, var, p 

if __name__ == '__main__':

	# format to use GP
	ker = Kernel()
	ker.SE(1,1)

	t = np.array([1,2,3,4])
	v = np.sin(t)

	predict_points = np.array([5])
	gp = GP()
	print gp.GPR(time_points = t, values = v,predict_point = predict_points, kernel = ker)



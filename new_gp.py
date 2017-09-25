from new_kernels import Kernel

import numpy as np 

from scipy import optimize
from scipy.linalg import solve
from scipy.spatial.distance import pdist, squareform,euclidean

from matplotlib import pyplot as plt

import pdb

class GP:
	def __init__(self, time_points, values, kernels, noise_level = 1.):
		
		self.time_points = time_points
		self.values = values

		self.noise_level = noise_level
		
		self.kernels = kernels

		self.cov = self.get_cov(self.kernels)
		self.margin_likelihood = self.get_margin_likelihood(self.cov)

	def get_cov(self, kernels):

		X = self.time_points

		N = len(X)
		K = np.zeros((N,N)) * 1.

		for i in range(len(kernels)):
			K += self.cal_K(kernels[i])

		return K

	def get_margin_likelihood(self, cov_matrix):
		
		if self.cov is None:
			self.cov = self.get_cov()

		X = self.time_points
		Y = self.values
		N = len(X)
		K = cov_matrix
		
		noise_level = 1.

		C = K + np.power(noise_level, 2) * np.identity(N)
		L = np.log(np.linalg.det(C))*0.5 + 0.5 * np.dot(np.dot(Y.T, np.linalg.inv(C)), Y) +  N * 1. / 2 * np.log(2* np.pi)

		return L

	def cal_K(self, kernel):
		
		X = self.time_points

		if kernel.kernel_type == "SE":
			R = (X.T - X)/kernel._pars[0]
			R = np.power(R,2)
			K = np.power(kernel._pars[1],2) * np.exp(-R)
		
		return K

	def optimization_function(self,parameters):

		X = self.time_points
		Y = self.values
		
		N = len(X)
		K = np.zeros((N,N)) * 1.

		new_kernels = []
		num_k = len(self.kernels)

		j = 0 
		for i in range(num_k):
			k_type = self.kernels[i].kernel_type
			if k_type == 'SE':
				k_par = (parameters[j:j+2])
				j = j + 2

			new_kernel = Kernel(k_type, k_par)
			new_kernels.append(new_kernel)


		noise_level = parameters[-1]
	
		K = self.get_cov(new_kernels)
		L = self.get_margin_likelihood(K)
		
		return L
		

	def optimize(self):

		parameters = []
		for kernel in self.kernels:
			parameters.append(kernel._pars) 

		parameters.append([self.noise_level])
		parameters = np.ravel(parameters)
		print parameters
		optimize.fmin_bfgs(self.optimization_function, parameters)


		
if __name__ == '__main__':

	X = np.array([1,2,3,4]).reshape(-1,1)
	Y = np.sin(X) + np.random.randn(4,1)

	k1 = Kernel("SE",1,1)

	gp1 = GP(time_points = X, values = Y, kernels =[k1,k1,k1,k1])
	gp1.optimize()



	# the optimization results should be the same but the parameters should be different for GPy and new_gp because GP using SE in a different form , so the value of the scaler should be different 

from new_kernels import Kernel

import numpy as np 

from scipy import optimize
from scipy.linalg import solve
from scipy.spatial.distance import pdist, squareform,euclidean

from matplotlib import pyplot as plt
import GPy
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
		
		C = K + np.power(self.noise_level, 2) * np.identity(N)
		L = np.log(np.linalg.det(C))*0.5 + 0.5 * np.dot(np.dot(Y.T, np.linalg.inv(C)), Y) +  N * 1. / 2 * np.log(2* np.pi)

		return L

	def cal_K(self, kernel):
		
		X = self.time_points

		if kernel.kernel_type == "SE":
			R = (X.T - X)/kernel.pars[0]
			R = np.power(R,2)
			K = np.power(kernel.pars[1],2) * np.exp(-R)
		
		return K


		#optimize.fmin_bfgs(self.optimization_function, parameters)

	def optimize(self):
		origin_paramters = []
		
		for ker in self.kernels:
			origin_paramters += ker.pars

		origin_paramters += [self.noise_level]

		new_parameters = optimize.fmin_bfgs(self.opt_marginal_likelihood, origin_paramters)

		new_kernels = []
		j = 0
		
		for i in range(len(self.kernels)):
			k_type = self.kernels[i].kernel_type 
			if k_type == 'SE':
				k_par = (new_parameters[j],new_parameters[j+1])
				j += 2
			new_k = Kernel(k_type, *k_par)
			new_kernels.append(new_k)

		self.kernels = new_kernels
		self.noise_level = new_parameters[j]


	def opt_marginal_likelihood(self, parameters):
		new_kernels = []
		j = 0
		for i in range(len(self.kernels)):
			k_type = self.kernels[i].kernel_type 
			if k_type == 'SE':
				k_par = (parameters[j],parameters[j+1])
				j += 2
			new_k = Kernel(k_type, *k_par)
			new_kernels.append(new_k)

		noise_level = parameters[j]

		X = self.time_points
		Y = self.values
		N = len(X)
	
		K = self.get_cov(new_kernels)
		
		C = K + np.power(noise_level, 2) * np.identity(N)
		L = np.log(np.linalg.det(C))*0.5 + 0.5 * np.dot(np.dot(Y.T, np.linalg.inv(C)), Y) +  N * 1. / 2 * np.log(2* np.pi)

		return L

	def display(self):
		print 'Kernels'
		for i in range(len(self.kernels)):
			self.kernels[i].display()

		print 'Noise level:'
		print self.noise_level

		
if __name__ == '__main__':

	X = np.array([1,2,3,4]).reshape(-1,1)
	Y = np.sin(X) + np.random.randn(4,1)

	k1 = Kernel("SE",1,1)

	gp1 = GP(time_points = X, values = Y, kernels =[k1])

	gp1.optimize()
	gp1.display()

	kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1)
	m = GPy.models.GPRegression(X,Y,kernel)

	m.optimize()






	# the optimization results should be the same but the parameters should be different for GPy and new_gp because GP using SE in a different form , so the value of the scaler should be different 
	print m

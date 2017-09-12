import numpy as np 
import scipy 
from scipy import optimize
from new_kernels import Kernel

from matplotlib import pyplot as plt
from scipy.linalg import solve
from scipy.spatial.distance import pdist, squareform,euclidean

from matplotlib import pyplot as plt


class GP:
	def __init__(self, time_points, values, kernels):
		
		self.time_points = time_points
		self.values = values
		
		self.kernels = []
		self.kernels += kernels

		self.kernel_type_list = []
		self.kernel_parameter_list = []
		
		for k in kernels:
			self.kernel_type_list.append(k.kernel_type)
			self.kernel_parameter_list.append(k.pars)

		self.cov = self.get_cov()
		self.margin_likelihood = self.get_margin_likelihood()

	def get_cov(self):

		X = self.time_points

		N = len(X)
		K = np.zeros((N,N)) * 1.

		for ker in self.kernels:
			K += ker.cal_K(X)

		return K

	def get_margin_likelihood(self):
		
		if self.cov is None:
			self.cov = self.get_cov()

		X = self.time_points
		Y = self.values
		N = len(X)
		K = self.cov
		
		noise_level = 1.

		C = K + np.power(noise_level, 2) * np.identity(N)
		L = np.log(np.linalg.det(C))*0.5 + 0.5 * np.dot(np.dot(Y.T, np.linalg.inv(C)), Y) +  N * 1. / 2 * np.log(2* np.pi)

		return L


	def optimization_function(self,parameters):

		X = self.time_points
		Y = self.values
		
		N = len(X)
		K = np.zeros((N,N)) * 1.

		noise_level = parameters[-1]

		j = 0

		for i in range(len(self.kernels)):
			ker = self.kernels[i]
			par_num = ker.par_num
			ker.pars = parameters[j:j+par_num]
			j = j + par_num
			K += ker.cal_K(X)

		C = K + np.power(noise_level, 2) * np.identity(N)
		L = np.log(np.linalg.det(C))*0.5 + 0.5 * np.dot(np.dot(Y.T, np.linalg.inv(C)), Y) +  N * 1. / 2 * np.log(2* np.pi)
		return -L
		
	def optimize(self):

		pars = self.kernel_parameter_list
		flat_pars = [item for sublist in pars for item in sublist]
		flat_pars.append(1.)
		print scipy.optimize.fmin_bfgs(self.optimization_function, flat_pars)

if __name__ == '__main__':

	X = np.array([1,2,3,4])
	Y = np.sin(X) # np.random.randn(20,1)*0.05

	k1 = Kernel("SE",1,1)
	#k2 = Kernel("SE",1,1)
	
	gp1 = GP(time_points = X, values = Y, kernels =[k1])

	gp1.optimize()

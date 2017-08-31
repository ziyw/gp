# Gaussian process regression 
# GPR 
import numpy as np 
import scipy 
from kernels import Kernel
from matplotlib import pyplot as plt
from scipy.linalg import solve
from scipy.spatial.distance import pdist, squareform,euclidean
from matplotlib import pyplot as plt

# for test, delete latter 
import GPy

class GP:
	def __init__(self, time_points, values, kernel, mean = 0):
		
		self.time_points = time_points
		self.values = values 

		self.N = time_points.size # number of time points 

		self.mean = mean 
		self.kernel = kernel

		if kernel.type == "SE":
			kernel.cal_SE(time_points)
		

	def get_likelihood(self, time_point, value):
		return self.GPR(time_point, value, self.kernel)
		
	def plot(self):

		range_min = np.min(self.time_points)
		range_max = np.max(self.time_points)

		axis_x = np.arange(range_min-1,range_max+1,0.1)
		fig = plt.figure(0)

		#plt.axis([0,5,-2,2], facecolor = 'g')
		plt.grid(color='w', linestyle='-', linewidth=0.5)

		ax = fig.add_subplot(111)
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.patch.set_facecolor('#E8E8F1')

		# show mean 
		mu = np.zeros(axis_x.size)
		var = np.zeros(axis_x.size)
		
		for i in range(axis_x.size):
			mu[i],var[i],_ = self.GPR(predict_point = axis_x[i], kernel = ker)
		
		plt.fill_between(axis_x,mu + var,mu-var,color = '#D1D9F0')
		
		# show mean 
		plt.plot(axis_x, mu, linewidth = 2, color = "#5B8CEB")
		# show the points
		plt.scatter(self.time_points, self.values,color = '#598BEB')
		plt.show()

	def test():
		"""
		A test function to compare GP with GPY
		"""
		pass 

	# optimize() auto gradient 

	# a way compare GPs
	def plot_compare(self,other_GPs):
		# other_GPs should be a vector of GPs
		# each one of them should be showed in the plot
		# mu and scatter should be in different colors 

		pass

	def GPR(self,predict_point,kernel, noise_level = 1):
		# how to adding kernels together?
		
		time_points = self.time_points
		values = self.values

		print kernel.K

		X = np.append(time_points, predict_point).reshape(-1,1)
		N = time_points.size
		K = kernel.K



		cov_K = K

		cov_k_K,cov_k = kernel.cal_new_SE(self.time_points, predict_point)
		print cov_k_K 
		print cov_k
		print cov_K

		# need to add the noise level later
		s = noise_level 
		A = cov_K + np.identity(N) * s * s
		
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
	ker = Kernel("SE",1,1)

	t = np.array([1,2,3,4])
	v = np.sin(t)

	predict_points = np.array([5])
	
	gp = GP(time_points = t, values = v, kernel = ker, mean = 0)
	print gp.GPR(predict_point = predict_points,kernel = ker, noise_level = 1)
	gp.plot()
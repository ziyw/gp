# Gaussian process regression 
# GPR 
import numpy as np 
import scipy 
from kernels import Kernel
from matplotlib import pyplot as plt
from scipy.linalg import solve
from scipy.spatial.distance import pdist, squareform,euclidean
from matplotlib import pyplot as plt

# for
import GPy

class GP:
	def __init__(self, time_points, values, kernel, mean = 0):
		
		self.time_points = time_points
		self.values = values 

		self.N = time_points.size # number of time points 

		self.mean = mean 
		self.kers = []
		self.kers.append(kernel)


		if kernel.type == "SE":
			kernel.cal_SE(time_points.reshape(-1,1))
		
		self.num_kernel = len(self.kers)


	def get_likelihood(self, time_point, value):
		return self.GPR(time_point, value, self.kernel)
		
	def plot(self):
		'''
		Plot the Gaussian Process object itself
		'''
		range_min = np.min(self.time_points)
		range_max = np.max(self.time_points)

		axis_x = np.arange(range_min-1,range_max+1,0.1)
		fig = plt.figure(0)


		# background color
		#plt.axis([0,5,-2,2], facecolor = 'g')
		# plt.grid(color='w', linestyle='-', linewidth=0.5)
		# ax.patch.set_facecolor('#E8E8F1')
		
		ax = fig.add_subplot(111)
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)

		# show mean 
		mu = np.zeros(axis_x.size)
		var = np.zeros(axis_x.size)
		
		for i in range(axis_x.size):
			mu[i],var[i],_ = self.GPR(predict_point = axis_x[i], kernel = ker)
		
		plt.fill_between(axis_x,mu + var,mu-var,color = '#D1D9F0')
		
		# show mean 
		plt.plot(axis_x, mu, linewidth = 2, color = "#5B8CEB")
		plt.xlabel('Time points')
		plt.ylabel('Values')
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
		
		if self.num_kernel == 1:
			K = self.kers[0].K
			cov_K = K
			ker = self.kers[0]
			cov_k_K,cov_k = ker.cal_new_SE(self.time_points, predict_point)
		
		else :

			K = self.kers[0].K
			ker = self.kers[0]
			cov_k_K,cov_k = ker.cal_new_SE(self.time_points, predict_point)

			for i in range(1,self.num_kernel):
				K += self.kers[i].K
				ker = self.kers[i]
				k1,k2 = ker.cal_new_SE(self.time_points, predict_point)
				cov_k_K += k1
				cov_k += k2
		
		cov_K = K

		time_points = self.time_points
		values = self.values

		N = time_points.size

		# need to add the noise level later
		# noise level is another kernel adding up 
		# so s actually could be removed 
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

	def add_kernel(self, new_kernel):
		self.kers.append(new_kernel)

		if new_kernel.type == "SE":
			new_kernel.cal_SE(self.time_points.reshape(-1,1))
		
		self.num_kernel = len(self.kers)
		pass

	def set_kernels(self, new_kernels):
		'''
		Set all kernels to a new set 
		'''
		pass
if __name__ == '__main__':

	# format to use GP
	ker = Kernel("SE",1,1)

	t = np.array([1,2,3,4])
	v = np.sin(t) + np.random.normal(0,1,4)

	predict_points = np.array([5])
	
	gp = GP(time_points = t, values = v, kernel = ker, mean = 0)
	gp.plot()
	print gp.GPR(predict_point = predict_points,kernel = ker, noise_level = 1)
	
	k2 = Kernel("SE",2,3)
	gp.add_kernel(k2)

	print gp.GPR(predict_point = predict_points,kernel = ker, noise_level = 1)
	# gp.plot()
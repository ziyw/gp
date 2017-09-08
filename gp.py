# Gaussian process regression 
# GPR 
import numpy as np 
import scipy 
from scipy import optimize
from kernels import Kernel
from matplotlib import pyplot as plt
from scipy.linalg import solve
from scipy.spatial.distance import pdist, squareform,euclidean

from matplotlib import pyplot as plt


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


	##############
	# optimization 
	##############

	def get_se_cov(self, x):
		h = x[0] * 1.
		l = x[1] * 1.

		X = np.matrix(self.time_points * 1.)

		R = (X.T - X)/l
		R = np.power(R, 2)
		K = np.power(h,2) * np.exp(-R)

		return K

	def se_function(self, x):

		output_scale = x[0]
		length_scale = x[1]
		noise_level = x[2]

		X = self.time_points
		Y = self.values

		K = self.get_se_cov(x)
		(N,_) = K.shape

		C = np.matrix(K + np.power(noise_level, 2) * np.identity(N) )

		L = np.log(np.linalg.det(C))*0.5 + 0.5 * np.dot(np.dot(Y.T, np.linalg.inv(C)), Y) +  N * 1. / 2 * np.log(2* np.pi)
		
		return L

	def optimize(self, kernel = 'SE'):

		self.parameters = scipy.optimize.fmin_bfgs(self.se_function, [1,1,1])
		print self.parameters


	def GPR(self,predict_point,noise_level = 1):

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
		# likelihood is wrong
		p = - 1.0 / 2.0 * np.dot(y , alpha) - np.sum(np.log(L.diagonal())) - (N/2.0 *  np.log (2 * np.pi))

		print p
		return mean, var, p 



	###########################
	# kernel part 
	##############

	# kernel manipulation 
	# sum add_kernel 
	# product product_kernel
	# def get_kernel(method = ["add", "product"])
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


	########
	# Making plot part 
	##########

	def plot(self):
		'''
		Plot the Gaussian Process object itself
		'''
		range_min = np.min(self.time_points)
		range_max = np.max(self.time_points)

		axis_x = np.arange(range_min-1,range_max+1,0.1)
		fig = plt.figure(0)

		# background color
		# plt.axis([0,5,-2,2], facecolor = 'g')
		# plt.grid(color='w', linestyle='-', linewidth=0.5)
		# ax.patch.set_facecolor('#E8E8F1')
		
		ax = fig.add_subplot(111)
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)

		# show mean 
		mu = np.zeros(axis_x.size)
		var = np.zeros(axis_x.size)
		
		for i in range(axis_x.size):
			mu[i],var[i],_ = self.GPR(predict_point = axis_x[i])
		
		plt.fill_between(axis_x,mu + var,mu-var,color = '#D1D9F0')
		
		# show mean 
		plt.plot(axis_x, mu, linewidth = 2, color = "#5B8CEB")
		plt.xlabel('Time points')
		plt.ylabel('Values')
		# show the points
		plt.scatter(self.time_points, self.values,color = '#598BEB', marker = 'X')
		plt.show()

	def plot_compare(self,other_GPs):
		# other_GPs should be a vector of GPs
		# each one of them should be showed in the plot
		# mu and scatter should be in different colors 
		# change to different scatter 
		GPs = [self, other_GPs]

		plt.xlabel('Time points')
		plt.ylabel('Values')
		mean_color = ["#0E746B",'#7CAB49','#F49331']
		cov_color = ['#6BC7C1','#E5E672','#E5E672']

		k_color = 3

		# find the max range
		range_min = np.min(self.time_points)
		range_max = np.max(self.time_points)

		for i in range(len(GPs)):
			tmin = np.min(GPs[i].time_points)
			tmax = np.max(GPs[i].time_points)
			range_min = min(range_min, tmin)
			range_max = max(range_max, tmax)
		
		axis_x = np.arange(range_min-1,range_max+1,0.1)
		fig = plt.figure(0)
		ax = fig.add_subplot(111)
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)

		for i in range(len(GPs)):

			# show mean 
			mu = np.zeros(axis_x.size)
			var = np.zeros(axis_x.size)
			
			for j in range(axis_x.size):
				mu[j],var[j],_ = GPs[i].GPR(predict_point = axis_x[j])
			 
			# print mu 
			# print var
			plt.fill_between(axis_x,mu + var,mu-var,color = cov_color[i % k_color],alpha = 0.5)
			
			# show mean 
			plt.plot(axis_x, mu, linewidth = 2, color = mean_color[i % k_color])


		# background color
		# plt.axis([0,5,-2,2], facecolor = 'g')
		# plt.grid(color='w', linestyle='-', linewidth=0.5)
		# ax.patch.set_facecolor('#E8E8F1')
		# show the points
		for i in range(len(GPs)):
			plt.scatter(GPs[i].time_points, GPs[i].values,color = mean_color[i % k_color], marker = 'X')
		
		plt.show()


if __name__ == '__main__':
	
	X = np.matrix([1,2,3,4]).reshape(-1,1)
	Y = np.sin(X) # np.random.randn(20,1)*0.05
	X = X.reshape(4,)
	Y = Y.reshape(4,)

	k1 = Kernel("SE",np.sqrt(2),1)
	
	gp1 = GP(time_points = X.T, values = Y.T, kernel = k1)
	gp1.optimize()
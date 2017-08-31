import numpy as np 

class Kernel:

	def __init__(self, kernel_type, para1, para2, *args):
		
		self.type = kernel_type 
		self.K = np.identity(1)

		if kernel_type == "SE":
			self.output_scale = para1
			self.input_scale = para2
		
		# if kernel_type == "RQ":
		# 	self.output_scale = 1
		# 	self.input_scale = 1
		# 	self.index = 1

		# if kernel_type == "per_SE":
		# 	self.output_scale = 1
		# 	self.time_period = 1
		# 	self.length_scale = 1

	def SE(self,output_scale,input_scale):
		self.output_scale = output_scale
		self.input_scale = input_scale
		self.type = 'SE'

	def cal_SE(self,X):
		'''
		Calculate SE covariance matrix
		'''
		X = X * 1.
		h = self.output_scale
		l = self.input_scale

		R = (X.T - X)/l
		R = np.power(R, 2)
		K = np.power(h,2) * np.exp(-R)
		
		self.K = K
		return K

	def cal_new_SE(self, X, new_point):
		# return K(k*)  and K(k*,K)
		X = X * 1.
		N = X.size

		h = self.output_scale
		l = self.input_scale

		R = (X - new_point)/ l
		R = np.power(R, 2)
		K = np.power(h,2) * np.exp(-R)
		
		cov_k_K = K[0:N]
		cov_k = K[-1]
		# because it is different for every points, no need to save in the object
		return cov_k_K,cov_k

	def RQ(self, output_scale, input_scale, index):
		self.output_scale = output_scale
		self.input_scale = input_scale 
		self.index = index 

	def cal_RQ(self,X):
		X = X * 1.
		h = self.output_scale
		alpha = self.index
		l = self.input_scale

		R = np.power(X.T - X,2)
		R = R / (alpha * l * l)
		R = h*h * (R + 1)
		
		K = np.power(R, -alpha)

		self.K = K
		return K

	def per_SE(self, output_scale, time_period, length_scale):
		self.output_scale = output_scale
		self.time_period = time_period
		self.length_scale = length_scale

	def cal_per_SE(self,X):
		X = X * 1.
		h = self.output_scale 
		T = self.time_period
		w = self.length_scale
		
		R = np.pi * np.abs((X.T - X)/ T)
		R = np.sin(R)
		R = (1. / (2 * w * w)) * np.power(R, 2)
		K = np.power(h,2) * np.exp(-R)
		
		return K 

if __name__ == '__main__':
	x = np.array([1,2,3,4]).reshape(-1,1)
	ker = Kernel("SE",1,1)
	print ker.cal_SE(x)

	# ker.RQ(1,2,3)
	# print ker.cal_RQ(x)

	# ker.per_SE(1,2,3)
	# print ker.cal_per_SE(x)
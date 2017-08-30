import numpy as np 

class Kernel:

	def SE(self,output_scale,input_scale):
		self.output_scale = output_scale
		self.input_scale = input_scale

	def cal_SE(self,X):
		'''
		Calculate SE covariance matrix
		Including K(x,x*), and K(x*,x*)
		'''
		h = self.output_scale
		l = self.input_scale

		R = (X.T - X)/ l
		R = np.power(R, 2)
		K = np.power(h,2) * np.exp(-R)
		
		self.K = K
		return K

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
	ker = Kernel()
	
	ker.SE(1,1)
	print ker.cal_SE(x)

	ker.RQ(1,2,3)
	print ker.cal_RQ(x)


	ker.per_SE(1,2,3)
	print ker.cal_per_SE(x)
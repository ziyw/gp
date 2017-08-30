import numpy as np 

class Kernel:

	@classmethod
	def SE(cls,output_scale,input_scale):
		'''
		parameters for 
		'''
		Kernel.type = 'SE'
		Kernel.output_scale = output_scale
		Kernel.input_scale = input_scale
		#self.type = "SE"

		#self.output_scale = output_scale
	
	@classmethod
	def cal_SE(cls,X):
		'''
		Calculate SE covariance matrix
		Including K(x,x*), and K(x*,x*)
		'''
		h = Kernel.output_scale
		l = Kernel.input_scale

		R = (X.T - X)/ l
		R = np.power(R, 2)
		
		K = np.power(h,2) * np.exp(-R)
		
		self.K = K
		return K

		pass 

	def RQ(self, output_scale, input_scale, index):
		pass 

	def cal_RQ(self,X):
		pass 

	def per_SE(self, output_scale, time_period, length_scale):
		pass 

	def cal_per_SE(self,X):
		pass 


	#def RQ(self, h, alpha, l):
		"""
		Rational quadratic 
		input arguments:
			h: output amplitude 
			alpha : index 
			l : input scale 
		output:	
			h^2(1 + (x_i - x_j) ^2 / (alpha * l^2))^(-alpha)
		"""
		self.h = h
		self.alpha = alpha 
		self.l = l

		X = self.time
		R = np.power(X.T - X,2)
		R = R / (alpha * l * l)
		R = h*h * (R + 1)
		
		K = np.power(R, -alpha)

		self.K = K
		return K

	##def per_SE(self, h,w,l):

		X = self.time
		R = np.pi * np.abs((X.T - X)/ l)
		R = np.sin(R)
		R = (1. / (2 * w * w)) * np.power(R, 2)
		K = np.power(h,2) * np.exp(-R)
		self.K = K

		return K

# ker = Kernel.SE(1,1)


if __name__ == '__main__':
	ker = Kernel.SE(1,1)
	print ker.cal_SE([1,2,3,4])

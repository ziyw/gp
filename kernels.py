import numpy as np 

class Kernel:

	def __init__(self,T):
		# T time points
		self.T = np.array(X, dtype = 'float64') 
		N = T.size 
		self.K = np.zeros((N,N))

	def SE(self,h,l):
		'''
		input arguments:
			h: output-scale amplitude 
			l: input scale 
		output: 
			h^2( exp [- ((x_i - x_j) / l)^2 ]) 
		'''
		self.type = "SE"
		X = self.T

		R = (X.T - X)/ l
		R = np.power(R, 2)
		
		K = np.power(h,2) * np.exp(-R)
		
		self.K = K
		return K

	def RQ(self, h, alpha, l):
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

		X = self.T
		R = np.power(X.T - X,2)
		R = R / (alpha * l * l)
		R = h*h * (R + 1)
		
		K = np.power(R, -alpha)

		self.K = K
		return K

	def white_noise(self,sigma):
		"""
		input arguments:
			sigma: white noise parameter
		output :
			sigma ^ 2 x delta(i,j)
		"""	
		X = self.T
		self.sigma = sigma
		K = np.identity(N)
		K = K * sigma * sigma 

		return K

	def per_SE(self, h,w,l):

		X = self.T
		R = np.pi * np.abs((X.T - X)/ l)
		R = np.sin(R)
		R = (1. / (2 * w * w)) * np.power(R, 2)
		K = np.power(h,2) * np.exp(-R)
		self.K = K

		return K



if __name__ == '__main__':

	X = np.array([1,2,3,4]).reshape(-1,1)
	k = Kernel(X)
	k.per_SE(1,2,3)

	print k.K 

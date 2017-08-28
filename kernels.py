import numpy as np 

class Kernel:

	def __init__(self,X,N):
		# X is a array of scaler 
		self.X = np.array(X, dtype = 'float64') 
		self.N = N
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
		X = self.X

		R = (X.T - X)/ l
		R = np.power(R, 2)

		K = h*h * np.exp(-R)

		self.K = K

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

		X = self.X
		R = np.power(X.T - X,2)
		R = R / (alpha * l * l)
		R = h*h * (R + 1)
		
		K = np.power(R, -alpha)

		self.K = K

	def white_noise(self,sigma):
		"""
		input arguments:
			sigma: white noise parameter
		output :
			sigma ^ 2 x delta(i,j)
		"""	
		X = self.X
		self.sigma = sigma
		K = np.identity(N)
		K = K * sigma * sigma 

		return K



if __name__ == '__main__':

	X = np.array([1,2,3,4]).reshape(-1,1)
	k = Kernel(X,4)
	k.RQ(1,2,3)

	print k.K 

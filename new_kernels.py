import numpy as np 

class Kernel:
	def __init__(self, kernel_type, *pars):
		
		self.kernel_type = kernel_type
		
		if kernel_type == 'SE':
			self.par_num = 2
			self._pars = list(pars)

	@property 
	def pars(self):
		return self._pars
	
	@pars.setter
	def pars(self, values):
		if self.kernel_type == 'SE':
			print "Set new value to kernels"
			self._pars = list(values)


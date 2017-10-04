import numpy as np 

class Kernel:
	def __init__(self, kernel_type, *pars):
		
		self.kernel_type = kernel_type
		
		if kernel_type == 'SE':
			self.par_num = 2
			self.pars = pars

	def display(self):
		print self.kernel_type
		print self.pars
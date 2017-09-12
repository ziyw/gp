import numpy as np 

class Kernel:
	def __init__(self, kernel_type, par1, par2, *args):
		
		self.kernel_type = kernel_type
		
		if kernel_type == 'SE':
			self.par_num = 2
			self.h = par1 * 1.
			self.l = par2 * 1.
			self.pars = [par1,par2]

	@property 
	def pars(self):
		return self.pars
	
	@pars.setter
	def pars(self, values):
		
		if self.kernel_type == 'SE':
			self.h = values[1]
			self.l = values[2]
			self.pars = values


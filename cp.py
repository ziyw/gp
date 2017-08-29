'''
Compare the results with gpy
'''
import numpy as np
import GPy

k = GPy.kern.RBF(1,1,1)
X = np.array([1,2,3,4]).reshape(-1,1)
mu = np.zeros((500))
C = k.K(X,X)
print C

from kernels import Kernel 
from gp import GP

k = Kernel(X)
k.SE(1,np.sqrt(2))
print k.K

# the kernel function is  the same 





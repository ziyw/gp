'''
Compare the results with gpy
Compare the two with plot
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


time = np.random.random(30)*5
value = np.sin(time) + np.random.normal(0,1,30)

# plot the result 
axis_x = np.arange(0,5.1,0.1)
plt.figure(0)
plt.axis([0,5,-2,2])
plt.fill_between(axis_x,-1,1)
plt.scatter(time, value,color = 'r')

plt.show()

# 	# X = np.linspace(0.05,0.95,10)[:,None]
# 	t = np.array([1,2,3,4])
# 	value = np.sin(t)
# 	x_test = np.matrix([5])

# 	gp = GP(X,y)
# 	print gp.regression(x_test,"RQ", 4,5,6)


# 	plt.figure(0)
# 	kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
#     + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
# 	gp = GaussianProcessRegressor(kernel=kernel,
#                               alpha=0.0).fit(X, y)
# X_ = np.linspace(0, 5, 100)
# y_mean, y_cov = gp.predict(X_[:, np.newaxis], return_cov=True)
	
# 	default_mean = 0.
# 	default_cov = np.ones(N)
# 	plt.plot(t, value)
# 	plt.fill_between(t, value)

# plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
# plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
#                  y_mean + np.sqrt(np.diag(y_cov)),
#                  alpha=0.5, color='k')
# plt.plot(X_, 0.5*np.sin(3*X_), 'r', lw=3, zorder=9)
# plt.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
# plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
#           % (kernel, gp.kernel_,
#              gp.log_marginal_likelihood(gp.kernel_.theta)))
# plt.tight_layout()



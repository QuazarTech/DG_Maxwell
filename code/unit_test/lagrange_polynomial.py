from app import lagrange
from utils import utils
import math
import arrayfire as af
import numpy as np
from matplotlib import pyplot as plt

from app import global_variables as gvar

plt.rcParams['figure.figsize'] = 9.6, 6.
plt.rcParams['figure.dpi'] = 100
plt.rcParams['image.cmap'] = 'jet'
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 20
plt.rcParams['font.sans-serif'] = 'serif'
plt.rcParams['text.usetex'] = False
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['xtick.major.pad'] = 8
plt.rcParams['xtick.minor.pad'] = 8
plt.rcParams['xtick.color'] = 'k'
plt.rcParams['xtick.labelsize'] = 'medium'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['ytick.major.pad'] = 8
plt.rcParams['ytick.minor.pad'] = 8
plt.rcParams['ytick.color'] = 'k'
plt.rcParams['ytick.labelsize'] = 'medium'
plt.rcParams['ytick.direction'] = 'in'

def lagrangePolynomialTest():
	'''
	A test function which plots the L1 norm of error against the number of LGL
	points taken.
	'''
	L1_norm = af.interop.np_to_af_array(np.zeros([15]))
	
	for n in range (2,16):
		gvar.populateGlobalVariables(n)
		
		y_LGL = af.arith.sin(2 * math.pi * gvar.xi_LGL)
		
		#random = -1 + 2 * af.random.randu(100, dtype = af.Dtype.f64)
		#x_random = af.algorithm.sort(random)
		x_random      = utils.linspace(-1, 1, 50)
		
		index         = af.range(gvar.N_LGL)
		Basis_array   = lagrange.lagrange_basis(index, x_random)
		y_interpolate = af.transpose(af.blas.matmul(af.transpose(y_LGL),\
										Basis_array))
		y_analytical  =  af.arith.sin(2 * math.pi * x_random)
		error         = y_interpolate - y_analytical
		
		L1_norm[(n - 2)] = af.sum(af.abs(error))
	
	number_LGL_Nodes = utils.linspace(2, 16, 15)
	
	plt.title(r'$L_1$ norm of error vs N')
	plt.xlabel(r'$N_{LGL}$')
	plt.ylabel(r'$L_1$ Norm.')
	
	plt.loglog(number_LGL_Nodes, L1_norm, basex = 2)
	plt.loglog(number_LGL_Nodes , (number_LGL_Nodes / 9.2) **\
				(-number_LGL_Nodes * 1.1),basex = 2)
	
	plt.legend(['L_1 norm', r'$(\frac{N}{9.3})^{-1.1N}$'])
	
	plt.show()
	
	return

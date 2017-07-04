#~ /usr/bin/env python3

import arrayfire as af
import numpy as np
af.set_backend('opencl')
from app import lagrange
from utils import utils
from app import global_variables as gvar
def polulateGlobalVariables():
	'''
	'''
	gvar.N_LGL       = 9
	gvar.xi_LGL      = lagrange.LGL_points(gvar.N_LGL)
	
	gvar.lBasisArray = af.interop.np_to_af_array( \
		lagrange.lagrange_basis_coeffs(gvar.xi_LGL))
	
	return


def test():
	'''
	'''
	
	i = 5
	x = utils.linspace(-1, 1, 5)
	
	#while True:
	#	lagrange.lagrange_basis(i, x).shape
	
	return


if __name__ == '__main__':
	'''
	'''
	af.info()
	
	polulateGlobalVariables()
	
	test()
	
	x = utils.linspace(-1,1,10)
	print(lagrange.lagrange_basis(2,x))
	
	
	#x_nodes = af.range(gvar.xi_LGL.shape[0])
	
	#basis_fn_value_array = lagrange.lagrange_basis(x_nodes, gvar.xi_LGL)
	
	#sin_LGL = af.transpose(af.arith.sin(gvar.xi_LGL))
	
	#print(sin_LGL,af.blas.matmul(sin_LGL, basis_fn_value_array))
	
	
	pass



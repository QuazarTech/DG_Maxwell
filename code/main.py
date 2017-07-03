#~ /usr/bin/env python3

import arrayfire as af

from app import lagrange
from utils import utils
from app import global_variables as gvar

def polulateGlobalVariables():
	'''
	'''
	gvar.N_LGL       = 16
	gvar.xi_LGL      = lagrange.LGL_points(gvar.N_LGL)
	
	gvar.lBasisArray = af.interop.np_to_af_array( \
		lagrange.lagrange_basis_coeffs(gvar.xi_LGL))
	
	return


def test():
	'''
	'''
	
	i = 5
	x = utils.linspace(-1, 1, 5)
	
	while True:
		lagrange.lagrange_basis(i, x).shape
	
	return


if __name__ == '__main__':
	'''
	'''
	af.set_backend('cuda')
	af.info()
	
	polulateGlobalVariables()
	
	test()
	
	pass

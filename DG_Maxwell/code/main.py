#~ /usr/bin/env python3

import arrayfire as af
import math
import numpy as np
from matplotlib import pyplot as plt
from app import lagrange
from utils import utils
from app import global_variables as gvar

def populateGlobalVariables(k):
	'''
	'''
	gvar.N_LGL       = k
	gvar.xi_LGL      = lagrange.LGL_points(gvar.N_LGL)
	
	gvar.lBasisArray = af.interop.np_to_af_array( \
		lagrange.lagrange_basis_coeffs(gvar.xi_LGL))
	
	return

def test():
	'''
	'''
	
	i = 5
	x = utils.linspace(-1, 1, 5)
	
	return

def lagrangePolynomialTest():
    
    L1_norm = af.interop.np_to_af_array(np.zeros([15]))
    for i in range (2,16):
        populateGlobalVariables(i)
        
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
        
        L1_norm[(i - 2)] = af.sum(af.abs(error))
    
    number_LGL_Nodes = utils.linspace(2, 16, 15)
    
    
    plt.loglog(number_LGL_Nodes, L1_norm, basex = 2)
    plt.loglog(number_LGL_Nodes , (number_LGL_Nodes/9.5) ** (-number_LGL_Nodes),
               basex = 2)
                     #The data points are fitted by a curve, (N/9.5)^(-N).
                     #Where N is the number of LGL points.
    plt.show()
    
   
    
if __name__ == '__main__':
	'''
	'''
	af.set_backend('opencl')
	af.info()
	
	populateGlobalVariables(16)
	
	test()
	
	lagrangePolynomialTest()
	
	pass



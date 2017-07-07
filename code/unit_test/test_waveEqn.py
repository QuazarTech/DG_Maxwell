import numpy as np
import arrayfire as af
af.set_backend('opencl')
from app import lagrange
from app import global_variables as gvar
from scipy import special as sp
from app import wave_equation
def test_lobatto_weight_function():
	threshold = 1e-14
	
	check_n3 =  np.sum(np.abs(gvar.lobatto_weight_function(3, \
		lagrange.LGL_points(3))-np.array([1/3, 4/3, 1/3]))) <= threshold
	
	check_n4 = np.sum(np.abs(gvar.lobatto_weight_function(4, \
		lagrange.LGL_points(4))-np.array([1/6, 5/6, 5/6, 1/6]))) <= threshold
	
	check_n5 = np.sum(np.abs(gvar.lobatto_weight_function(5, \
		lagrange.LGL_points(5))-np.array([0.1, 49/90, 32/45, 49/90, 0.1]))) <= \
			threshold 
	
	assert check_n3 & check_n4 & check_n5
	
	
def test_Lp_Lj_xi():
	threshold = 1e-14
	a = np.zeros([3,3,3])
	a[0] = np.array([[1,2,3],[8,10,12],[21,24,27]])
	a[1] = np.array([[4,8,12],[20,25,30],[42,48,54]])
	a[2] = np.array([[7,14,21],[32,40,48],[63,72,81]])
	analytical_product = af.interop.np_to_af_array(a)
	test_array = af.interop.np_to_af_array(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype = np.float64))
	test_array1 = af.reorder(test_array, 2, 0, 1)
	check_order3 = af.sum(af.abs(wave_equation.Lj_Lp_xi(test_array, test_array1) - analytical_product)) <= threshold
	assert check_order3
	

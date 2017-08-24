import numpy as np
import arrayfire as af
import math
from matplotlib import pyplot as plt
from app import lagrange
from app import global_variables as gvar
from app import wave_equation
from utils import utils
af.set_backend('opencl')




def test_isoparam_x():
	'''
	
	'''
	threshold = 1e-14
	test_nodes = af.interop.np_to_af_array(np.array([[0., 2], [0, 1], [0, 0], \
													[1, 0], [2, 0], [2, 1],\
													[2, 2], [1, 2.]]))
	
	xi  = af.Array([0])
	eta = af.Array([0])
	x_nodes = test_nodes[:, 0]
	y_nodes = test_nodes[:, 1]
	analytical_coordinate   = af.Array([1, 1])
	calculated_x_coordinate = wave_equation.isoparam_x(x_nodes, xi, eta)
	calculated_y_coordinate = wave_equation.isoparam_y(y_nodes, xi, eta)
	calculated_coordinate   = af.Array([af.sum(calculated_x_coordinate),\
										af.sum(calculated_y_coordinate)])

	assert(af.sum(calculated_coordinate - analytical_coordinate) <= threshold)


def test_jacobian():
	'''
	Taking an element with nodes (0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (2, 1)
	(2, 2), (1, 2). We expect the jacobian everywhere in the element to to be 1 
	it is just transposed by a distance.
	'''
	threshold = 7e-12
	
	test_nodes = af.interop.np_to_af_array(np.array([[0, 2], [0, 1], [0, 0], \
													[1, 0], [2, 0], [2, 1],\
													[2, 2], [1, 2]]))
	xi  = af.interop.np_to_af_array(np.array([0.355]))
	eta = af.interop.np_to_af_array(np.array([0.64580]))
	
	x_nodes = test_nodes[:, 0]
	y_nodes = test_nodes[:, 1]
	
	analytical_dx_dxi = wave_equation.dx_dxi_numerical(x_nodes, xi, eta)
	analytical_dy_dxi = wave_equation.dy_dxi_numerical(y_nodes, xi, eta)
	analytical_dx_deta = wave_equation.dx_deta_numerical(x_nodes, xi, eta)
	analytical_dy_deta = wave_equation.dy_deta_numerical(y_nodes, xi, eta)	
	calculated_jacobian = wave_equation.jacobian(x_nodes, y_nodes, xi, eta)
	analytical_jacobian = 1
	
	assert(abs(af.sum(calculated_jacobian) - analytical_jacobian )<= threshold)

def test_lobatto_weights():
	'''
	'''
	
	gvar.populateGlobalVariables(3)
	threshold = 1e-14
	calculated_weights = (gvar.lobatto_weights)
	analytical_weights = af.interop.np_to_af_array(np.array([1/3, 4/3, 1/3]))
	print(gvar.N_LGL)
	diff = (2 - af.sum(calculated_weights))
	print(diff)
	
	assert diff < threshold

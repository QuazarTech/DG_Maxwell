import numpy as np
import arrayfire as af
af.set_backend('opencl')
from app import lagrange
from app import global_variables as gvar
from app import wave_equation
from matplotlib import pyplot as plt
from utils import utils
import math

def test_lobatto_weight_function():
	'''
	Test function to check the lobatto weights for known LGL points.
	'''
	threshold = 1e-14

	check_n3 =  np.sum(np.abs(gvar.lobatto_weight_function(3, \
		lagrange.LGL_points(3))-np.array([1/3, 4/3, 1/3]))) <= threshold
	
	check_n4 = np.sum(np.abs(gvar.lobatto_weight_function(4, \
		lagrange.LGL_points(4))-np.array([1/6, 5/6, 5/6, 1/6]))) <= threshold
	
	check_n5 = np.sum(np.abs(gvar.lobatto_weight_function(5, \
		lagrange.LGL_points(5))-np.array([0.1, 49/90, 32/45, 49/90, 0.1]))) <= \
			threshold
	
	assert check_n3 & check_n4 & check_n5


def test_Li_Lp_xi():
	'''
	A Test function to check the Li_Lp_xi function in wave_equation module by
	passing two test arrays and comparing the analytical product with the
	numerically calculated one with a tolerance of 1e-14.
	'''
	
	gvar.populateGlobalVariables(3)
	
	threshold = 1e-14
	
	test_array = af.interop.np_to_af_array(np.array([[1, 2, 3],  \
													 [4, 5, 6],  \
													 [7, 8, 9]], \
										   dtype = np.float64))
	
	test_array1 = af.reorder(test_array, 2, 0, 1)
	
	product = np.zeros([3,3,3])
	product[0] = np.array([[1,2,3],[8,10,12],[21,24,27]])
	product[1] = np.array([[4,8,12],[20,25,30],[42,48,54]])
	product[2] = np.array([[7,14,21],[32,40,48],[63,72,81]])
	
	analytical_product = af.interop.np_to_af_array(product)
	
	check_order3 = af.sum(af.abs(wave_equation.Li_Lp_xi(test_array, test_array1)
							  - analytical_product)) <= threshold
	
	assert check_order3


def test_dx_dxi():
	'''
	A Test function to check the dx_xi function in wave_equation module by
	passing nodes of an element and using the LGL points. Analytically, the
	differential would be a constant. The check has a tolerance 1e-7.
	'''
	threshold = 1e-7
	nodes = np.array([7, 10], dtype = np.float64)
	test_nodes = af.interop.np_to_af_array(nodes)
	analytical_dx_dxi = 1.5
	check_dx_dxi = af.sum(af.abs(wave_equation.dx_dxi_numerical(test_nodes,
																gvar.xi_LGL)
							  - analytical_dx_dxi)) <= threshold
	
	assert check_dx_dxi


def test_dx_dxi_analytical():
	'''
	Test to check the dx_dxi_analytical in wave equation module for an element
	and compare it with an analytical value
	'''
	threshold = 1e-14
	nodes = af.Array([2,6])
	check_analytical_dx_dxi = af.sum(af.abs(wave_equation.dx_dxi_analytical
										 (nodes, 0) - 2)) <= threshold
	assert check_analytical_dx_dxi


def test_A_matrix():
	'''
	Test function to check the A_matrix function in wave_equation module.
	
	Obtaining the A_matrix from the function and setting the value of
	all elements above a certain threshold to be 1 and plotting it.
	'''
	
	gvar.populateGlobalVariables(8)
	threshold          = 1e-5
	A_matrix_structure = np.zeros([gvar.N_LGL, gvar.N_LGL])
	non_zero_indices   = np.where(np.array(wave_equation.A_matrix()) > threshold)
	
	A_matrix_structure[non_zero_indices] = 1.
	
	plt.gca().invert_yaxis()
	plt.contourf(A_matrix_structure, 100, cmap = 'Blues')
	plt.axes().set_aspect('equal')
	plt.colorbar()
	plt.show()

	return

def test_lBasisArray():
	'''
	Function to test the lBasisArray function in global_variables module by
	passing 8 LGL points and comparing the numerically obtained basis function
	coefficients to analytically calculated ones.
	'''
	threshold = 1e-12
	gvar.populateGlobalVariables(8)
	basis_array_analytical = np.zeros([8, 8])
	
	basis_array_analytical[0] = np.array([-3.351562500008004,\
										3.351562500008006, \
										3.867187500010295,\
										-3.867187500010297,\
										- 1.054687500002225, \
										1.054687500002225, \
										0.03906249999993106,\
										- 0.03906249999993102])
	basis_array_analytical[1] = np.array([8.140722718246403,\
										- 7.096594831382852,\
										- 11.34747768400062,\
										9.89205188146461, \
										3.331608712119162, \
										- 2.904297073479968,\
										- 0.1248537463649464,\
										0.1088400233982081])
	basis_array_analytical[2] = np.array([-10.35813682892759,\
										6.128911440984293,\
										18.68335515838398,\
										- 11.05494463699297,\
										- 8.670037141196786,\
										5.130062549476987,\
										0.3448188117404021,\
										- 0.2040293534683072])

	basis_array_analytical[3] = np.array([11.38981374849497,\
										- 2.383879109609436,\
										- 24.03296250200938,\
										5.030080255538657,\
										15.67350804691132,\
										- 3.28045297599924,\
										- 3.030359293396907,\
										0.6342518300700298])

	basis_array_analytical[4] = np.array([-11.38981374849497,\
										- 2.383879109609437,\
										24.03296250200939,\
										5.030080255538648,\
										- 15.67350804691132,\
										- 3.28045297599924,\
										3.030359293396907,\
										0.6342518300700299])

	basis_array_analytical[5] = np.array([10.35813682892759,\
										6.128911440984293,\
										-18.68335515838398,\
										- 11.05494463699297,\
										8.670037141196786,\
										5.130062549476987,\
										- 0.3448188117404021,\
										- 0.2040293534683072])
	basis_array_analytical[6] = np.array([-8.140722718246403,\
										- 7.096594831382852,\
										11.34747768400062,\
										9.89205188146461, \
										-3.331608712119162, \
										- 2.904297073479968,\
										0.1248537463649464,\
										0.1088400233982081])
	basis_array_analytical[7] = np.array([3.351562500008004,\
										3.351562500008005, \
										- 3.867187500010295,\
										- 3.867187500010298,\
										1.054687500002225, \
										1.054687500002224, \
										- 0.039062499999931,\
										- 0.03906249999993102])
				
	basis_array_analytical = af.interop.np_to_af_array(basis_array_analytical)
	
	assert af.sum(af.abs(basis_array_analytical - gvar.lBasisArray)) < threshold
	
def test_lobatto_quadrature():
	'''
	Test function to check if lobatto quadrature method gives an answer within
	a specified tolerance.
	'''
	threshold = 1e-10
	N = 8
	gvar.populateGlobalVariables(N)
	
	y_LGL = (gvar.xi_LGL ** (10))
	lobatto_integral = af.sum(y_LGL * af.interop.np_to_af_array( \
		gvar.lobatto_weight_function(gvar.N_LGL, gvar.xi_LGL)))
	
	analytical_integral = 2 / 11
	check_lobatto       = (lobatto_integral- analytical_integral) <= threshold
	
	print(y_LGL, af.interop.np_to_af_array( \
		gvar.lobatto_weight_function(gvar.N_LGL, gvar.xi_LGL)))
	
	assert check_lobatto


def test_gaussian_weights():
	'''
	Test function to check the Gaussian_weights function in the global 
	global_variables
	module
	
	Note
	----
	The accuracy of the gaussian weight is only 1e-7. This causes accuracy 
	errors in the A matrix / Integral_Li_Lp calculation.
	'''
	
	gvar.populateGlobalVariables(5)
	threshold = 1e-7
	N = 5
	gaussian_weights = np.zeros([N])
	
	for i in range (0, N):
		gaussian_weights[i] = gvar.gaussian_weights(N, i)
	
	reference_weights = np.array([0.23692688505618908, 0.47862867049936647,
					0.5688888888888, 0.47862867049936647, 0.23692688505618908 ])
	
	assert np.sum(np.abs(gaussian_weights - reference_weights)) <= threshold


def test_gauss_A_matrix():
	'''
	Calculates the value of lagrange basis functions obtained for :math: `N_LGL`
	points at the gaussian nodes.
	
	Returns
	-------
	The value of integral of product of lagrange basis functions with limits
	-1 and 1.
	'''
	
	N = 8
	gvar.populateGlobalVariables(8)
	
	x_tile           = af.transpose(af.tile(gvar.gauss_nodes, 1, N))
	power            = utils.linspace(N - 1, 0, N)
	power_tile       = af.tile(power, 1, N)
	x_pow            = af.arith.pow(x_tile, power_tile)
	L_1              = af.blas.matmul(gvar.lBasisArray[1], x_pow)
	L_0              = af.blas.matmul(gvar.lBasisArray[0], x_pow)
	gauss_weights    = af.np_to_af_array(np.zeros([N]))
	
	for i in range(0, N):
		gauss_weights[i] = gvar.gaussian_weights(N, i)
	
	Integral_L_0_0        = af.transpose(gauss_weights) * L_0 * L_0
	gaussian_weights_tile = af.tile(af.reorder(gauss_weights, 1, 2, 0), N, N)
	
	index = af.range(N)
	L_i   = af.blas.matmul(gvar.lBasisArray[index], x_pow)
	L_j   = af.reorder(L_i, 0, 2, 1)
	L_i   = af.reorder(L_i, 2, 0, 1)
	
	dx_dxi      = wave_equation.dx_dxi_numerical(af.transpose(gvar.x_nodes),\
																	gvar.xi_LGL)
	dx_dxi_tile = af.tile(af.reorder(dx_dxi, 1, 2, 0), gvar.N_LGL, gvar.N_LGL)
	
	L_array         = wave_equation.Li_Lp_xi(L_j, L_i)
	L_element       = (L_array * gaussian_weights_tile * dx_dxi_tile)
	Integral_Li_Lp  = af.sum(L_element, dim = 2)
	
	return Integral_Li_Lp

def test_Integral_Li_Lp():
	'''
	Test function to check the A_matrix function in wave_equation module.
	
	Obtaining the A_matrix from the function and setting the value of
	all elements above a certain threshold to be 1 and plotting it.
	'''
	
	gvar.populateGlobalVariables(8)
	threshold          = 1e-5
	A_matrix_structure = np.zeros([gvar.N_LGL, gvar.N_LGL])
	non_zero_indices   = np.where(np.array(test_gauss_A_matrix()) > threshold)
	
	A_matrix_structure[non_zero_indices] = 1.
	
	plt.gca().invert_yaxis()
	plt.contourf(A_matrix_structure, 100, cmap = 'Blues')
	plt.axes().set_aspect('equal')
	plt.colorbar()
	plt.show()
	
	return

def test_gauss_A_matrix_numerical():
	'''
	'''
	threshold = 1e-7
	
	reference_A_matrix = af.interop.np_to_af_array(np.array([\
	[0.03333333333332194, 0.005783175201965206, -0.007358427761753982, \
	0.008091331778355441, -0.008091331778233877, 0.007358427761705623, \
	-0.00578317520224949, 0.002380952380963754], \
	
	[0.005783175201965206, 0.19665727866729804, 0.017873132323192046,\
	-0.01965330750343234, 0.019653307503020866, -0.017873132322725152,\
	0.014046948476303067, -0.005783175202249493], \
	
	[-0.007358427761753982, 0.017873132323192046, 0.31838117965137114, \
	0.025006581762566073, -0.025006581761945083, 0.022741512832051156,\
	-0.017873132322725152, 0.007358427761705624], \
	
	[0.008091331778355441, -0.01965330750343234, 0.025006581762566073, \
	0.3849615416814164, 0.027497252976343693, -0.025006581761945083, \
	0.019653307503020863, -0.008091331778233875],  
	
	[-0.008091331778233877, 0.019653307503020866, -0.025006581761945083, \
	0.027497252976343693, 0.3849615416814164, 0.025006581762566073, \
	-0.019653307503432346, 0.008091331778355443], \
	
	[0.007358427761705623, -0.017873132322725152, 0.022741512832051156, \
	-0.025006581761945083, 0.025006581762566073, 0.31838117965137114, \
	0.017873132323192046, -0.0073584277617539835], \
	
	[-0.005783175202249493, 0.014046948476303067, -0.017873132322725152, \
	0.019653307503020863, -0.019653307503432346, 0.017873132323192046, \
	0.19665727866729804, 0.0057831752019652065], \
	
	[0.002380952380963754, -0.005783175202249493, 0.007358427761705624, \
	-0.008091331778233875, 0.008091331778355443, -0.0073584277617539835, \
	0.0057831752019652065, 0.033333333333321946]
	]))
	
	test_A_matrix = test_gauss_A_matrix()
	error_array = af.abs(reference_A_matrix - test_A_matrix)
	
	assert af.algorithm.max(error_array) < threshold
	
def test_gaussian_quadrature():
	'''
	A test function to check the accuracy of gaussian quadrature and plotting
	the L1 norm of error against the number of gaussian nodes used.
	'''
	L1_norm = af.interop.np_to_af_array(np.zeros([29]))
	
	for N in range(2, 30):
		gvar.populateGaussNodes(N)
		gaussian_nodes = gvar.gauss_nodes
		gauss_weights = np.zeros([N])
		
		for i in range(0,N):
			gauss_weights[i]  = gvar.gaussian_weights(N, i)
			pass
		
		function_nodes = af.arith.sin(2 * math.pi * gaussian_nodes) + \
			15 * (gaussian_nodes ** 14)
		
		# Used a `x ^ 14` term since the product of lagrange basis polynomials
		# for 8 LGL points produces a `x ^ 14` term.
		
		numerical_integral  = af.sum(af.interop.np_to_af_array(gauss_weights)\
									* function_nodes)
		
		analytical_integral = 2  #Since integral of sin(2 * \\pi * x) is zero 
								 #and is two for `15 * x ^ 14` over -1 to 1.
		
		L1_norm[(N - 2)]    = abs(numerical_integral - analytical_integral)
		pass
	
	number_gaussian_Nodes = utils.linspace(2, 30, 29)
	plt.title(r'$L_1$ norm of error vs N')
	plt.xlabel(r'$no.nodes$')
	plt.ylabel(r'$L_1$ Norm.')
	
	plt.loglog(number_gaussian_Nodes, L1_norm, basex = 2)
	#plt.loglog(number_gaussian_Nodes , (number_gaussian_Nodes / 9.2) **\
		#(-number_gaussian_Nodes * 1.1),basex = 2)
	plt.legend(['L_1 norm', r'$(\frac{N}{9.3})^{-1.1N}$'])
	
	plt.show()
	
	return

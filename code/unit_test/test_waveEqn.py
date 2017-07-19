import numpy as np
import arrayfire as af
af.set_backend('opencl')
from app import lagrange
from app import global_variables as gvar
from app import wave_equation
from matplotlib import pyplot as plt
from utils import utils
import math

def test_Li_Lp_x_gauss():
	'''
	A Test function to check the Li_Lp_x_gauss function in wave_equation module
	by passing two test arrays and comparing the analytical product with the
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
	
	check_order3 = af.sum(af.abs(wave_equation.Li_Lp_x_gauss\
		(test_array, test_array1) - analytical_product)) <= threshold
	
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
	threshold = 1e-7
	gvar.populateGlobalVariables(5, 5)
	gaussian_weights = gvar.gauss_weights
	
	reference_weights = af.Array([0.23692688505618908, 0.47862867049936647,
					0.5688888888888, 0.47862867049936647, 0.23692688505618908 ])
	
	assert af.max(af.abs(gaussian_weights - reference_weights)) <= threshold


def test_Integral_Li_Lp():
	'''
	Test function to check the A_matrix function in wave_equation module.
	
	Obtaining the A_matrix from the function and setting the value of
	all elements above a certain threshold to be 1 and plotting it.
	'''
	threshold          = 1e-5
	gvar.populateGlobalVariables(8, 8)
	A_matrix_structure = np.zeros([gvar.N_LGL, gvar.N_LGL])
	non_zero_indices  = np.where(np.array(wave_equation.A_matrix()) > threshold)
	
	A_matrix_structure[non_zero_indices] = 1.
	
	plt.gca().invert_yaxis()
	plt.contourf(A_matrix_structure, 100, cmap = 'Blues')
	plt.axes().set_aspect('equal')
	plt.colorbar()
	plt.show()
	
	return


def test_A_matrix():
	'''
	Test function to check the A matrix obtained from wave_equation module with
	one obtained by numerical integral solvers.
	'''
	threshold = 1e-8
	gvar.populateGlobalVariables(8, 8)
	
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
	
	test_A_matrix = wave_equation.A_matrix()
	print(test_A_matrix.shape, reference_A_matrix.shape)
	error_array = af.abs(reference_A_matrix - test_A_matrix)
	
	assert af.algorithm.max(error_array) < threshold


def test_gaussian_quadrature():
	'''
	A test function to check the accuracy of gaussian quadrature and plotting
	the error against the number of gaussian nodes used.
	'''
	error = af.interop.np_to_af_array(np.zeros([29]))
	gvar.populateGlobalVariables(8)
	for N in range(2, 30):
		gvar.populateGlobalVariables(8, N)
		gaussian_nodes = gvar.gauss_nodes
		gauss_weights = gvar.gauss_weights
		
		function_nodes = af.arith.sinh(gaussian_nodes)
		
		polynomial_L_0          = np.poly1d(np.array(gvar.lBasisArray[0])[0])
		value_at_gaussian_nodes = af.interop.np_to_af_array\
			(polynomial_L_0(gaussian_nodes) ** 2 + np.array(function_nodes))
		Integral_L_0_gaussian   = value_at_gaussian_nodes * gauss_weights
		
		numerical_integral  = np.sum(Integral_L_0_gaussian)
		reference_integral  = 0.03333333333332194 #zero for :math:`sinh(x)`
		
		error[(N - 2)]    = abs(numerical_integral - reference_integral)
		pass
	af.display(error, 14)
	number_gaussian_Nodes = utils.linspace(2, 30, 29)
	plt.title(r'Error vs N')
	plt.xlabel(r'Number of Gaussian nodes')
	plt.ylabel(r'Error')
	
	plt.loglog(number_gaussian_Nodes, error, basex = 2)

	plt.legend(['Error'])
	
	plt.show()
	
	return

#def test_d_Lp_x_gauss_xi():
	#'''
	#Test function to check the d_Lp_x_gauss_xi function in the wave_equation 
	#module	with a numerically obtained one.
	#'''
	#threshold = 1e-13
	#gvar.populateGlobalVariables(8, 9)
	#reference_d_Lp_xi = af.interop.np_to_af_array(np.array([\
	#[-14.0000000000226, -3.20991570302344,0.792476681323880,-0.372150435728984,\
	#0.243330712724289, -0.203284568901545,0.219957514771985,-0.500000000000000],
	
	#[18.9375986071129, 3.31499272476776e-11, -2.80647579473469,1.07894468878725\
	#,-0.661157350899271,0.537039586158262, -0.573565414940005,1.29768738831567],
	
	#[-7.56928981931106, 4.54358506455201, -6.49524878326702e-12, \
	#-2.37818723350641, 1.13535801687865, -0.845022556506714, 0.869448098330221,\
	#-1.94165942553537],
	
	#[4.29790816425547,-2.11206121431525,2.87551740597844,-1.18896004153157e-11,\
	#-2.38892435916370, 1.37278583181113, -1.29423205091574, 2.81018898925442],
	
	#[-2.81018898925442, 1.29423205091574, -1.37278583181113, 2.38892435916370, \
	#1.18892673484083e-11,-2.87551740597844, 2.11206121431525,-4.29790816425547],
	
	#[1.94165942553537, -0.869448098330221, 0.845022556506714,-1.13535801687865,\
	#2.37818723350641, 6.49524878326702e-12,-4.54358506455201,7.56928981931106],\
	
	#[-1.29768738831567, 0.573565414940005,-0.537039586158262,0.661157350899271,\
	#-1.07894468878725,2.80647579473469,-3.31498162253752e-11,-18.9375986071129],
	
	#[0.500000000000000,-0.219957514771985,0.203284568901545,-0.243330712724289,\
	#0.372150435728984, -0.792476681323880, 3.20991570302344, 14.0000000000226]
	#]))
	
	#assert(af.max(reference_d_Lp_xi - lagrange.d_Lp_x_gauss_xi())) < threshold

def test_volume_integral_flux():
	'''
	A test function to check the volume_integral_flux function in the
	wave_equation module.
	'''
	threshold = 1e-9
	#print(wave_equation.volume_integral_flux(gvar.u[0]))
	analytical_flux_integral = 0
	
	return

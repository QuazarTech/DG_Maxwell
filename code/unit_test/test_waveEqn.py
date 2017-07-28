import numpy as np
import arrayfire as af
af.set_backend('cuda')
from app import lagrange
from app import global_variables as gvar
from app import wave_equation
from matplotlib import pyplot as plt
from utils import utils
import math


def test_mappingXiToX():
	'''
	'''
	threshold = 1e-14
	gvar.populateGlobalVariables()
	
	test_element_nodes = af.interop.np_to_af_array(np.array([7, 11]))
	test_xi            = 0
	analytical_x_value = 9
	numerical_x_value  = wave_equation.mappingXiToX(test_element_nodes, test_xi)
	
	assert af.abs(analytical_x_value - numerical_x_value) <= threshold


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
	
	product    = np.zeros([3,3,3])
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
	gvar.populateGlobalVariables(8)
	nodes = np.array([7, 10], dtype = np.float64)
	test_nodes = af.interop.np_to_af_array(nodes)
	analytical_dx_dxi = 1.5
	
	check_dx_dxi = (af.statistics.mean(wave_equation.dx_dxi_numerical
					(test_nodes,gvar.xi_LGL)) - analytical_dx_dxi) <= threshold
	
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
	
	NOTE
	----
	A diagonal A-matrix is used as a reference in this unit test.
	The A matrix calculated analytically gives a different matrix.
	
	'''
	threshold = 1e-8
	gvar.populateGlobalVariables(8, 8)
	
	reference_A_matrix = af.tile(gvar.lobatto_weights, 1, gvar.N_LGL)\
		* af.identity(gvar.N_LGL, gvar.N_LGL, dtype = af.Dtype.f64)
	
	test_A_matrix = wave_equation.A_matrix()
	error_array   = af.abs(reference_A_matrix - test_A_matrix)
	
	assert af.algorithm.max(error_array) < threshold


def test_d_Lp_xi():
	'''
	Test function to check the d_Lp_xi function in the lagrange module with a
	numerically obtained one.
	'''
	threshold = 1e-13
	gvar.populateGlobalVariables(8)
	
	reference_d_Lp_xi = af.interop.np_to_af_array(np.array([\
	[-14.0000000000226, -3.20991570302344,0.792476681323880,-0.372150435728984,\
	0.243330712724289, -0.203284568901545,0.219957514771985,-0.500000000000000],
	
	[18.9375986071129, 3.31499272476776e-11, -2.80647579473469,1.07894468878725\
	,-0.661157350899271,0.537039586158262, -0.573565414940005,1.29768738831567],
	
	[-7.56928981931106, 4.54358506455201, -6.49524878326702e-12, \
	-2.37818723350641, 1.13535801687865, -0.845022556506714, 0.869448098330221,\
	-1.94165942553537],
	
	[4.29790816425547,-2.11206121431525,2.87551740597844,-1.18896004153157e-11,\
	-2.38892435916370, 1.37278583181113, -1.29423205091574, 2.81018898925442],
	
	[-2.81018898925442, 1.29423205091574, -1.37278583181113, 2.38892435916370, \
	1.18892673484083e-11,-2.87551740597844, 2.11206121431525,-4.29790816425547],
	
	[1.94165942553537, -0.869448098330221, 0.845022556506714,-1.13535801687865,\
	2.37818723350641, 6.49524878326702e-12,-4.54358506455201,7.56928981931106],\
	
	[-1.29768738831567, 0.573565414940005,-0.537039586158262,0.661157350899271,\
	-1.07894468878725,2.80647579473469,-3.31498162253752e-11,-18.9375986071129],
	
	[0.500000000000000,-0.219957514771985,0.203284568901545,-0.243330712724289,\
	0.372150435728984, -0.792476681323880, 3.20991570302344, 14.0000000000226]
	]))
	
	assert(af.max(reference_d_Lp_xi - lagrange.d_Lp_xi(gvar.xi_LGL))) < threshold


def test_volume_integral_flux():
	'''
	A test function to check the volume_integral_flux function in wave_equation
	module by analytically calculated Gauss-Lobatto quadrature.
	
	NOTE
	----
	The analytically obtained flux integral by Gauss-Lobatto quadrature
	doesn't match result obtained by numerically integrating the flux integral
	term.
	
	However by taking limits which aren't -1 and 1. The flux integral method
	has much improved precision.
	
	The second check in this test function uses a reference value obtained
	numerically by a solver with machine accuracy.
	'''
	threshold = 1e-10
	gvar.populateGlobalVariables(8)
	analytical_flux_integral = af.transpose(af.interop.np_to_af_array(np.array(\
				[-0.0243250104044395, 0.0445985586016178, -0.412943909240457, \
					-0.592576678843147, 0.592576678843147, 0.412943909240457,\
									-0.0445985586016178, 0.0243250104044395])))
	
	calculated_flux_integral = wave_equation.volume_integral_flux(gvar.xi_LGL\
		, af.reorder(np.e ** (-(gvar.xi_LGL) ** 2 / 0.4 ** 2), 1, 0, 2))
	
	check1 = (af.max(af.abs(analytical_flux_integral - calculated_flux_integral)) 
		< threshold)
	
	#Here, we use an element from -1 to 0.8 and compare the numerically obtained
	#result and the one returned by volume_integral_flux.
	
	
	element1_x_nodes = af.reorder(gvar.element_nodes[0 : 1], 1, 0, 2)
	flux_integral    = np.array(wave_equation.volume_integral_flux(element1_x_nodes\
											, gvar.u[0, :, 0]))
	
	numerical_flux_integral = np.array([-0.005293926590211267, 0.0010839010095961025, \
		0.005653446247246795, -0.0022480977450876714, 0.0013285326802573401, \
			-0.0008800498093180824, 0.0005769753858060879, \
				-0.00022078117828930213])
	
	check2 = np.max(np.abs(flux_integral - numerical_flux_integral)) < threshold
	
	assert (check1 & check2)

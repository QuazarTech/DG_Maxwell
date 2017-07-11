#! /usr/bin/env python3

import arrayfire as af

from app import lagrange
from utils import utils
from app import global_variables as gvar

def Li_Lp_xi(L_xi_i, L_xi_p):
	'''	
	Parameters
	----------
	L_xi_i : arrayfire.Array [1 N N 1]
			 A 2D array :math:`L_i` calculated at all the
			 LGL points :math:`\\xi_j`
	
	L_xi_p : arrayfire.Array [N 1 N 1]
			 A 2D array :math:`L_p` calculated at all the
			 LGL points :math:`\\xi_j`
	
	Returns
	-------
	Li_Lp_xi : arrayfire.Array [N N N 1]
			   Matrix of :math:`L_p(\\xi)L_i(\\xi)`
	
	'''
	Li_Lp_xi = af.bcast.broadcast(utils.multiply, L_xi_i, L_xi_p)
	
	return Li_Lp_xi


def mappingXiToX(x_nodes, xi):
	'''
	Parameters
	----------
	
	x_nodes : arrayfire.Array
			  Element nodes.
	
	xi      : np.float64
			  Value of xi in domain (-1, 1) which returns the corresponding
			  x value in the
	
	Returns
	-------
	X value in the element with given nodes and xi.
	'''
	N_0 = (1. - xi) / 2
	N_1 = (xi + 1.) / 2
	
	N0_x0 = af.bcast.broadcast(utils.multiply, N_0, x_nodes[0])
	N1_x1 = af.bcast.broadcast(utils.multiply, N_1, x_nodes[1])
	
	return N0_x0 + N1_x1


def dx_dxi_numerical(x_nodes, xi):
	'''
	Differential calculated by central differential method about xi using the
	mappingXiToX function.
	
	Parameters
	----------
	
	x_nodes : arrayfire.Array
			  Contains the nodes of elements
	
	xi		: float
			  Value of xi
	
	Returns
	-------
	Numerical value of differential of X w.r.t the given xi 
	'''
	dxi = 1e-7
	x2 = mappingXiToX(x_nodes, xi + dxi)
	x1 = mappingXiToX(x_nodes, xi - dxi)
	
	return (x2 - x1) / (2 * dxi)


def dx_dxi_analytical(x_nodes, xi):
	'''
	
	Parameters
	----------
	x_nodes : arrayfire.Array
			  An array containing the nodes of an element.
	
	Returns
	-------
	The analytical solution to the dx/dxi for an element.
	
	'''
	return((x_nodes[1] - x_nodes[0]) / 2)


def A_matrix():
	'''
	
	:math::
		A_matrix = \Sigma L_{i}(\\xi) L_{p}(\\xi) w_{j} \frac{dx}{d s\\xi}
	
	The A matrix depends on the product of lagrange basis functions at two
	different indices for xi LGL points, The differential of x w.r.t xi at the
	LGL points. Taking the sum of the resultant array along dimension 2 gives
	the required A matrix.
	
	Returns
	-------
	The A matrix.
	
	'''
	lobatto_weights = af.interop.np_to_af_array(gvar.lobatto_weight_function
											 (gvar.N_LGL, gvar.xi_LGL))
	
	index = af.range(gvar.N_LGL)
	L_xi_i = lagrange.lagrange_basis(index, gvar.xi_LGL)
	L_xi_p = af.reorder(L_xi_i, 1, 2, 0)
	L_xi_i = af.reorder(L_xi_i, 2, 0, 1)
	
	Li_Lp_xi_array = Li_Lp_xi(L_xi_i, L_xi_p)
	
	lobatto_weights_tile = af.tile(af.reorder(lobatto_weights, 1, 2, 0),
							   gvar.N_LGL, gvar.N_LGL)
	
	dx_dxi      = dx_dxi_numerical(af.transpose(gvar.x_nodes), gvar.xi_LGL)
	dx_dxi_tile = af.tile(af.reorder(dx_dxi, 1, 2, 0), gvar.N_LGL, gvar.N_LGL)
	A_matrix    = af.sum(Li_Lp_xi_array * lobatto_weights_tile * dx_dxi_tile,
				   dim = 2)
	
	return A_matrix

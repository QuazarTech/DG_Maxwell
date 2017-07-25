#! /usr/bin/env python3

import arrayfire as af
import numpy as np
from app import lagrange
from utils import utils
from app import global_variables as gvar

def Li_Lp_x_gauss(L_x_gauss_i, L_x_gauss_p):
	'''
	Parameters    [TODO] : Impose a condition such that ONLY required arrays can
						   be passed
	----------
	L_x_gauss_i : arrayfire.Array [1 N N 1]
				  A 2D array :math:`L_i` obtained at LGL points calculated at the
				  gaussian nodes :math:`N_Gauss`.
	
	L_x_gauss_p : arrayfire.Array [N 1 N 1]
				  A 2D array :math:`L_p` obtained at LGL points calculated at the
				  gaussian nodes :math:`N_Gauss`.
	
	Returns
	-------	
	Li_Lp_x_gauss : arrayfire.Array [N N N 1]
			   Matrix of :math:`L_p (N_Gauss) L_i (N_Gauss)`.
	
	'''
	Li_Lp_x_gauss = af.bcast.broadcast(utils.multiply, L_x_gauss_i, L_x_gauss_p)
	
	return Li_Lp_x_gauss


def mappingXiToX(x_nodes, xi):
	'''
	Parameters
	----------
	
	x_nodes : arrayfire.Array
			  Element nodes.
	
	xi      : numpy.float64
			  Value of :math: `xi` in domain (-1, 1) which returns the 
			  corresponding :math: `x` value in the element.
	
	Returns
	-------
	:math: `X` value in the element with given nodes and :math: `xi`.
	'''
	N_0 = (1 - xi) / 2
	N_1 = (1 + xi) / 2
	
	N0_x0 = af.bcast.broadcast(utils.multiply, N_0, x_nodes[0, :])
	N1_x1 = af.bcast.broadcast(utils.multiply, N_1, x_nodes[1, :])
	
	return N0_x0 + N1_x1


def dx_dxi_numerical(x_nodes, xi):
	'''
	Differential calculated by central differential method about :math: `xi`
	using the mappingXiToX function.
	
	Parameters
	----------
	
	x_nodes : arrayfire.Array
			  Contains the nodes of elements
	
	xi		: float
			  Value of xi
	
	Returns
	-------
	Numerical value of differential of :math: `X` w.r.t the given :math: `xi`. 
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
	The analytical solution to the \\frac{dx}{d\\xi} for an element.
	
	'''
	return((x_nodes[1] - x_nodes[0]) / 2)


def A_matrix():
	'''
	Calculates the value of lagrange basis functions obtained for :math: `N_LGL`
	points at the gaussian nodes.
	
	Returns
	-------
	The value of integral of product of lagrange basis functions obtained by LGL
	points, using gaussian quadrature method using :math: `n_gauss` points. 
	'''	
	
	x_tile           = af.transpose(af.tile(gvar.xi_LGL, 1, gvar.N_LGL))
	power            = af.flip(af.range(gvar.N_LGL))
	power_tile       = af.tile(power, 1, gvar.N_LGL)
	x_pow            = af.arith.pow(x_tile, power_tile)
	lobatto_weights  = gvar.lobatto_weights
	
	lobatto_weights_tile = af.tile(af.reorder(lobatto_weights, 1, 2, 0),\
												gvar.N_LGL, gvar.N_LGL)
	
	index = af.range(gvar.N_LGL)
	L_i   = af.blas.matmul(gvar.lBasisArray[index], x_pow)
	L_j   = af.reorder(L_i, 0, 2, 1)
	L_i   = af.reorder(L_i, 2, 0, 1)
	
	dx_dxi      = dx_dxi_numerical(af.transpose(gvar.x_nodes),gvar.xi_LGL)
	dx_dxi_tile = af.tile(dx_dxi, 1, gvar.N_LGL, gvar.N_LGL)
	
	Li_Lp_array     = Li_Lp_x_gauss(L_j, L_i)
	L_element       = (Li_Lp_array * lobatto_weights_tile * dx_dxi_tile)
	A_matrix        = af.sum(L_element, dim = 2)
	
	return A_matrix


def flux_x(u):
    """
    """
    return gvar.c * u

def volume_integral_flux(u):
	'''
	'''
	d_Lp_xi     = af.transpose(lagrange.d_Lp_xi())
	weight_tile = af.tile(gvar.lobatto_weights, 1, gvar.N_LGL)
	flux        = af.reorder(flux_x(u), 2, 1, 0)
	flux_u_tile = af.tile(flux, 1, gvar.N_LGL)
	print(weight_tile, d_Lp_xi, flux_u_tile)
	integral    = af.sum(weight_tile * d_Lp_xi * flux_u_tile, 0)
	
	return integral

def lax_friedrichs_flux(left_state, right_state, c_lax):
    """
    Function to calculate the lax friedrichs flux which depends on the flux
    on either side of the boundary and also has a stability inducing term??
    
    Parameters
    ----------
    [TODO]
    """
    return 0.5*((flux_x(left_state) + flux_x(right_state)) - c_lax * \
		(right_state - left_state))



def b_vector(u_n):
	'''
	'''
	int_u_ni_Lp_Li   = af.blas.matmul(A_matrix(), af.transpose(u_n))
	int_flux_dLp_dxi = volume_integral_flux(u_n)
	
	L_p = gvar.lagrange_basis_function()
	
	#surface_term = L_p[-1] * lax_friedrichs_flux(u[n, i, -1], u[n, i + 1, ])
	
	return

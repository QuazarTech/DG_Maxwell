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
	N_0 = (1. - xi) / 2
	N_1 = (xi + 1.) / 2
	
	N0_x0 = af.bcast.broadcast(utils.multiply, N_0, x_nodes[0])
	N1_x1 = af.bcast.broadcast(utils.multiply, N_1, x_nodes[1])
	
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
	
	x_tile           = af.transpose(af.tile(gvar.gauss_nodes, 1, gvar.N_LGL))
	power            = af.flip(af.range(gvar.N_LGL))
	power_tile       = af.tile(power, 1, gvar.N_Gauss)
	x_pow            = af.arith.pow(x_tile, power_tile)
	gauss_weights    = gvar.gauss_weights
	
	gaussian_weights_tile = af.tile(af.reorder(gauss_weights, 1, 2, 0),\
												gvar.N_LGL, gvar.N_LGL)
	
	index = af.range(gvar.N_LGL)
	L_i   = af.blas.matmul(gvar.lBasisArray[index], x_pow)
	L_j   = af.reorder(L_i, 0, 2, 1)
	L_i   = af.reorder(L_i, 2, 0, 1)
	
	dx_dxi      = dx_dxi_numerical(af.transpose(gvar.x_nodes),gvar.gauss_nodes)
	dx_dxi_tile = af.tile(af.reorder(dx_dxi, 1, 2, 0), gvar.N_LGL, gvar.N_LGL)
	
	Li_Lp_array     = Li_Lp_x_gauss(L_j, L_i)
	L_element       = (Li_Lp_array * gaussian_weights_tile * dx_dxi_tile)
	A_matrix        = af.sum(L_element, dim = 2)
	print(A_matrix)
	return A_matrix


def flux_x(u):
    """
    """
    return gvar.c * u

def volume_integral_flux(u):
	'''
	'''
	d_Lp_x_gauss_xi   = af.transpose(lagrange.d_Lp_x_gauss_xi())
	weight_tile       = af.tile(gvar.gauss_weights, 1, gvar.N_LGL)
	flux_u_tile       = af.tile(af.transpose(flux_x(u)), 1, gvar.N_LGL)
	integral          = af.sum(weight_tile * d_Lp_x_gauss_xi * flux_u_tile, 0)
	print(d_Lp_x_gauss_xi)
	
	return integral


def b_vector(u_n):
	'''
	'''
	u_previous    = af.blas.matmul(A_matrix, af.transpose(u_n))
	
	return


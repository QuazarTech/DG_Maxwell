#! /usr/bin/env python3

import arrayfire as af
import numpy as np
from app import lagrange
from utils import utils
from app import global_variables as gvar

def Li_Lp_xi(L_i_xi, L_p_xi):
	'''
	Parameters
	----------
	L_i_xi : arrayfire.Array [1 N N 1]
				  A 2D array :math:`L_i` obtained at LGL points calculated at the
				  LGL nodes :math:`N_LGL`.
	
	L_p_xi : arrayfire.Array [N 1 N 1]
			 A 2D array :math:`L_p` obtained at LGL points calculated at the
			 LGL nodes :math:`N_LGL`.
	
	Returns
	-------	
	Li_Lp_xi : arrayfire.Array [N N N 1]
			   Matrix of :math:`L_p (N_LGL) L_i (N_LGL)`.
	
	'''
	Li_Lp_xi = af.bcast.broadcast(utils.multiply, L_i_xi, L_p_xi)
	
	return Li_Lp_xi


def mappingXiToX(x_nodes, xi):
	'''
	Parameters
	----------
	
	x_nodes : arrayfire.Array
			  Element nodes.
	
	xi      : numpy.float64
			  Value of :math: `\\xi`coordinate for which the corresponding
			  :math: `x` coordinate is to be found.
.
	
	Returns
	-------
	N0_x0 + N1_x1 : arrayfire.Array
				    :math: `x` value in the element corresponding to
				    :math:`\\xi`.
	'''
	N_0 = (1 - xi) / 2
	N_1 = (1 + xi) / 2
	
	N0_x0 = af.bcast.broadcast(utils.multiply, N_0, x_nodes[0])
	N1_x1 = af.bcast.broadcast(utils.multiply, N_1, x_nodes[1])
	
	return N0_x0 + N1_x1


def dx_dxi_numerical(x_nodes, xi):
	'''
	Differential calculated by central differential method about :math: `\\xi`
	using the mappingXiToX function.
	
	Parameters
	----------
	
	x_nodes : arrayfire.Array
			  Contains the nodes of elements
	
	xi		: float
			  Value of :math: `\\xi`
	
	Returns
	-------
	(x2 - x1) / (2 * dxi) : arrayfire.Array
							:math:`\\frac{dx}{d \\xi}`. 
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
	(x_nodes[1] - x_nodes[0]) / 2 : arrayfire.Array
									The analytical solution of
									\\frac{dx}{d\\xi} for an element.
	
	'''
	return (x_nodes[1] - x_nodes[0]) / 2


def A_matrix():
	'''
	Calculates the value of lagrange basis functions obtained for :math: `N_LGL`
	points at the LGL nodes.
	
	Returns
	-------
	A_matrix : arrayfire.Array
			   The value of integral of product of lagrange basis functions
			   obtained by LGL points, using Gauss-Lobatto quadrature method
			   using :math: `N_LGL` points. 
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
	L_p   = af.reorder(L_i, 0, 2, 1)
	L_i   = af.reorder(L_i, 2, 0, 1)
	
	dx_dxi      = dx_dxi_numerical((gvar.x_nodes),gvar.xi_LGL)
	dx_dxi_tile = af.tile(dx_dxi, 1, gvar.N_LGL, gvar.N_LGL)
	Li_Lp_array     = Li_Lp_xi(L_p, L_i)
	L_element       = (Li_Lp_array * lobatto_weights_tile * dx_dxi_tile)
	A_matrix        = af.sum(L_element, dim = 2)
	
	return A_matrix


def flux_x(u):
    '''
    A function which returns the value of flux for a given wave function u.
    :math:`f(u) = c u^k`
    
    Parameters
    ----------
    u         : arrayfire.Array [N 1 1 1]
				A 1-D array which contains the value of wave function.
	
	Returns
	-------
	c * u : arrayfire.Array
			The value of the flux for given u.
    '''
    return gvar.c * u


def volumeIntegralFlux(element_nodes, u):
	'''
	A function to calculate the volume integral of flux in the wave equation.
	:math:`\\int_{-1}^1 f(u) \\frac{d L_p}{d\\xi} d\\xi`
	This will give N values of flux integral as p varies from 0 to N - 1.
	
	This integral is carried out over an element with LGL nodes mapped onto it.
	
	Parameters
	----------
	element_nodes : arrayfire.Array [N 1 1 1]
					A 1-D array consisting of the LGL nodes mapped onto the
					element's domain.
	
	u             : arrayfire.Array [1 N 1 1]
					A 1-D array containing the value of the wave function at the
					mapped LGL nodes in the element.
	
	Returns
	-------
	flux_integral : arrayfire.Array [1 N 1 1]
					A 1-D array of the value of the flux integral calculated
					for various lagrange basis functions.
	'''
	
	dLp_xi        = af.tile(af.transpose(gvar.dLp_xi), 1, 1, gvar.N_Elements)
	weight_tile   = af.tile(gvar.lobatto_weights, 1, gvar.N_LGL,\
																gvar.N_Elements)
	flux          = af.reorder(flux_x(u), 1, 2, 0)
	flux_u_tile   = af.tile(flux, 1, gvar.N_LGL, 1)
	flux_integral = af.sum((weight_tile * dLp_xi * flux_u_tile), 0)
	
	return af.reorder(flux_integral, 2, 1, 0)

def elementFluxIntegral(n):
	'''
	Function which reorders the element numbers which can then be passed
	into volumeIntegralFlux. 
	
	Parameters
	----------
	n  :  Element numbers for which the flux integral is to be calculated,
		  Passing an array of 0 to N_Elements - 1 would give the flux integral
		  for all elements at all :math:`p`
	
	Returns
	-------
	volumeIntegralFlux(element_n_x_nodes,\
						gvar.u[n, :, 0])  : arrayfire.Array
											An array of :math:`\\int_{-1}^1 f(u)
											\\frac{d L_p}{d\\xi} d\\xi` for all
											elements
	'''
	element_n_x_nodes = af.reorder(gvar.element_nodes[n], 1, 0, 2)
	
	return volumeIntegralFlux(element_n_x_nodes, gvar.u[n, :, 0])


def lax_friedrichs_flux(u, t_n):
	'''
	'''
		
	u_n_0              = u[1:, 0, t_n]
	u_nminus1_N_LGL    = u[:gvar.N_Elements, -1, t_n]
	flux_n_0           = flux_x(u_n_0)
	flux_nminus1_N_LGL = flux_x(u_nminus1_N_LGL)
	
	lax_friedrichs_flux = (flux_n_0 + flux_nminus1_N_LGL) / 2 \
						- gvar.c_lax * (u_n_0 - u_nminus1_N_LGL)
    
    return lax_friedrichs_flux

def b_vector(u_n):
	'''
	NOTE 
	Incomplete.
	'''
	int_u_ni_Lp_Li   = af.blas.matmul(A_matrix(), af.transpose(u_n))
	int_flux_dLp_dxi = volume_integral_flux(u_n)
	
	return

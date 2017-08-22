#! /usr/bin/env python3

from os import sys
import numpy as np
from matplotlib import pyplot as plt
import subprocess

import arrayfire as af
af.set_backend('opencl')
from tqdm import trange

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



def isoparam_x(x_nodes, xi, eta):
	"""
	Parameters
	----------
	x_nodes : af.Array
			  An array consisting of the x nodes of the element.
	
	xi      : numpy.float64
			  The :math: `\\xi` coordinate for which the `x` coordinate is to
			  be found
	
	eta     : numpy.float64
			  The :math: `\\eta` coordinate for which the `x` coordinate is to
			  be found.
	
	Returns
	-------
	isoparametric_map_x : The x coordinate obtained by mapping the xi-eta domain
	
	A function which maps the x coordinates of the nodes of elements onto a
	xi-eta domain with nodes at (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)
	, (1, 0), (1, 1) and (0, 1)
	"""
	N_2 = (-1.0 / 4.0) * (1 - xi)  * (1 - eta) * (1 + xi + eta)
	N_3 = (1.0 / 2.0)  * (1 - eta) * (1 - xi**2)
	N_4 = (-1.0 / 4.0) * (1 + xi)  * (1 - eta) * (1 - xi + eta)
	N_5 = (1.0 / 2.0)  * (1 + xi)  * (1 - eta**2)
	N_6 = (-1.0 / 4.0) * (1 + xi)  * (1 + eta) * (1 - xi - eta)
	N_7 = (1.0 / 2.0)  * (1 + eta) * (1 - xi**2)
	N_0 = (-1.0 / 4.0) * (1 - xi)  * (1 + eta) * (1 + xi - eta)
	N_1 = (1.0 / 2.0)  * (1 - xi)  * (1 - eta**2)
	
	isoparametric_map_x = N_0 * af.sum(x_nodes[0]) \
					+ N_1 * af.sum(x_nodes[1]) \
					+ N_2 * af.sum(x_nodes[2]) \
					+ N_3 * af.sum(x_nodes[3]) \
					+ N_4 * af.sum(x_nodes[4]) \
					+ N_5 * af.sum(x_nodes[5]) \
					+ N_6 * af.sum(x_nodes[6]) \
					+ N_7 * af.sum(x_nodes[7])
	return isoparametric_map_x


def isoparam_y(y_nodes, xi, eta):
	"""
	Parameters
	----------
	y_nodes : af.Array
			  An array consisting of the y nodes of the element.
	
	xi      : numpy.float64
			  The :math: `\\xi` coordinate for which the `x` coordinate is to
			  be found
	
	eta     : numpy.float64
			  The :math: `\\eta` coordinate for which the `x` coordinate is to
			  be found.
	
	Returns
	-------
	isoparametric_map_y : The y coordinate obtained by mapping the xi-eta domain
	
	
	A function which maps the y coordinates of the nodes of elements onto a
	xi-eta domain with nodes at (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)
	, (1, 0), (1, 1) and (0, 1)
	"""
	isoparametric_map_y = isoparam_x(y_nodes, xi, eta)
	
	return isoparametric_map_y


def dx_dxi_numerical(x_nodes, xi, eta):
	'''
	Differential calculated by central differential method about :math: `\\xi`
	using the isoparam_x function.
	
	Parameters
	----------
	
	x_nodes : arrayfire.Array
			  Contains the x coordinates of the element nodes.
	
	xi      : float
			  Value of :math: `\\xi`
	
	eta     : float
			  Value of :math: `\\eta`
	
	Returns
	-------
	(x2 - x1) / (2 * dxi) : arrayfire.Array
							:math:`\\frac{\\partial x}{\\partial \\xi}`. 
	'''
	dxi = 1e-4
	x2 = isoparam_x(x_nodes, xi + dxi, eta)
	x1 = isoparam_x(x_nodes, xi - dxi, eta)
	
	return (x2 - x1) / (2 * dxi)


def dy_dxi_numerical(y_nodes, xi, eta):
	'''
	Differential calculated by central differential method about :math: `\\xi`
	using the isoparam_y function.
	
	Parameters
	----------
	
	y_nodes : arrayfire.Array
			  Contains the y coordinates of the element nodes.
	
	xi      : float
			  Value of :math: `\\xi`
	
	eta     : float
			  Value of :math: `\\eta`
	
	Returns
	-------
	(y2 - y1) / (2 * dxi) : arrayfire.Array
							:math:`\\frac{\\partial y}{\\partial \\xi}`. 
	'''
	dxi = 1e-4
	y2 = isoparam_y(y_nodes, xi + dxi, eta)
	y1 = isoparam_y(y_nodes, xi - dxi, eta)
	
	return (y2 - y1) / (2 * dxi)


def dx_deta_numerical(x_nodes, xi, eta):
	'''
	Differential calculated by central differential method about :math: `\\eta`
	using the isoparam_x function.
	
	Parameters
	----------
	
	x_nodes : arrayfire.Array
			  Contains the x coordinates of the element nodes.
	
	xi      : float
			  Value of :math: `\\xi`
	
	eta     : float
			  Value of :math: `\\eta`
	
	Returns
	-------
	(x2 - x1) / (2 * dxi) : arrayfire.Array
							:math:`\frac{\\partial y}{\\partial \\eta}`. 
	'''
	deta = 1e-4
	x2   = isoparam_x(x_nodes, xi, eta + deta)
	x1   = isoparam_x(x_nodes, xi, eta - deta)
	
	return (x2 - x1) / (2 * deta)


def dy_deta_numerical(y_nodes, xi, eta):
	'''
	Differential calculated by central differential method about :math: `\\eta`
	using the isoparam_y function.
	
	Parameters
	----------
	
	y_nodes : arrayfire.Array
			  Contains the y coordinates of the element nodes.
	
	xi      : float
			  Value of :math: `\\xi`
	
	eta     : float
			  Value of :math: `\\eta`
	
	Returns
	-------
	(y2 - y1) / (2 * deta) : arrayfire.Array
							:math:`\\frac{\\partial x}{\\partial \\eta}`. 
	'''
	deta = 1e-4
	y2   = isoparam_y(y_nodes, xi, eta + deta)
	y1   = isoparam_y(y_nodes, xi, eta - deta)
	
	return (y2 - y1) / (2 * deta)


def jacobian(x_nodes, y_nodes, xi, eta):
	"""
	The Jacobian is given by :math:
	`\\frac{\\partial x}{\\partial \\xi} \\frac{\\partial y}{\\partial \\eta}
	 - \\frac{\\partial x}{\\partial \\eta} \\frac{\\partial y}{\\partial \\xi}`
	
	Parameters
	----------
	x_nodes : arrayfire.Array
			  Contains the x coordinates of the element nodes.
	
	y_nodes : arrayfire.Array
			  Contains the y coordinates of the element nodes.
	
	xi      : float
			  Value of :math: `\\xi`
	
	eta     : float
			  Value of :math: `\\eta`
	
	Returns
	-------
	jacobian : float
			   The determinant of the Jacobian matrix.
	
	"""
	
	dx_dxi  = dx_dxi_numerical (x_nodes, xi, eta)
	dy_deta = dy_deta_numerical(y_nodes, xi, eta)
	dx_deta = dx_deta_numerical(x_nodes, xi, eta)
	dy_dxi  = dy_dxi_numerical (y_nodes, xi, eta)
	jacobian = (dx_dxi * dy_deta) - (dx_deta * dy_dxi)
	
	return jacobian

def A_matrix():
	'''
	Calculates the A matrix.
	'''
	
	L_i         = lagrange.lagrange_basis_function()
	L_p         = af.reorder(L_i, 0, 2, 1)
	L_i         = af.reorder(L_i, 2, 0, 1)
	Li_Lp_array = (Li_Lp_xi(L_i, L_p))
	
	collapsed_Li_Lp_array_xi  = af.moddims(Li_Lp_array, d0 = gvar.N_LGL ** 2,\
																d2 = gvar.N_LGL)
	lobatto_weights           = af.reorder(gvar.lobatto_weights, 2, 1, 0)
	weight_collapsed_array_xi = af.bcast.broadcast(utils.multiply, \
									collapsed_Li_Lp_array_xi, lobatto_weights)
	tile_weight_array_xi      = af.tile(weight_collapsed_array_xi, gvar.N_LGL)
	tile_weight_array_xi      = af.moddims(tile_weight_array_xi,\
									d0 = gvar.N_LGL ** 2, d1 = gvar.N_LGL ** 2)
	
	
	weight_array_eta      = af.reorder(weight_collapsed_array_xi, 2, 0, 1)
	tile_weight_array_eta = af.tile(weight_array_eta, gvar.N_LGL)
	Li_Lp_Lq_Lj           = af.blas.matmul(tile_weight_array_xi,\
											tile_weight_array_eta)
	
	xi_LGL_tile     = af.flat(af.transpose(af.tile(gvar.xi_LGL, 1, gvar.N_LGL)))
	eta_LGL_tile    = af.tile(gvar.eta_LGL, gvar.N_LGL)
	jacobian_xi_eta = af.transpose(jacobian(gvar.x_nodes, gvar.y_nodes,\
													xi_LGL_tile, eta_LGL_tile))
	
	
	A_matrix = af.bcast.broadcast(utils.multiply, Li_Lp_Lq_Lj, jacobian_xi_eta)
	
	return A_matrix


def flux_x(u):
    '''
    A function which returns the value of flux for a given wave function u.
    :math:`f(u) = c u^k`
    
    Parameters
    ----------
    u         : arrayfire.Array
				A 1-D array which contains the value of wave function.
	
	Returns
	-------
	c * u : arrayfire.Array
			The value of the flux for given u.
    '''
    return gvar.c * u


def laxFriedrichsFlux():
	'''
	A function which calculates the lax-friedrichs_flux :math:`f_i` using.
	:math:`f_i = \\frac{F(u^{i + 1}_0) + F(u^i_{N_{LGL} - 1})}{2} - \frac
					{\Delta x}{2\Delta t} (u^{i + 1}_0 - u^i_{N_{LGL} - 1})`
	
	'''
	
	return


def b_vector():
	'''
	A function which returns the b vector for N_Elements number of elements.
	'''
	
	return


def time_evolution():
	'''
	Simulates the time evolution of a wave equation.
	'''
	
	return

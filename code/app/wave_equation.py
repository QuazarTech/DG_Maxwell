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


def mappingXiToX(x_nodes, xi):
	'''
	Function for isoparametric mapping from x coordinates to xi coordinates.
	'''
	
	return 


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
	Calculates the A matrix.
	'''
	
	return


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

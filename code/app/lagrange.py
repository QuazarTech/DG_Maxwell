#! /usr/bin/env python3

from os import sys

import arrayfire as af
af.set_backend('opencl')
import numpy as np
from utils import utils
from app import global_variables as gvar

def LGL_points(N):
	'''
	Returns the :math: `N` Legendre-Gauss-Laguere points which are
	the roots of the equation
	:math::
		(1 - x^2)L'_N = 0
	
	Parameters
	----------
	N : int
		Number of LGL-points to be generated.
		2 < N < 16
	
	Returns
	-------
	lgl : arrayfire.Array
		  An array of :math: `N` LGL points.
	'''
	if N > 16 or N < 2:
		print('Skipping! This function can only return from ',
			  '2 to 16 LGL points.')
	
	n = N - 2

	lgl = af.Array(gvar.LGL_list[n])
	return lgl


def lagrange_basis_coeffs(xi_LGL):    
	'''
	A function to get the coefficients of the Lagrange basis polynomials for
	a given set of x nodes.
	This function calculates the Lagrange basis
	polynomials by this formula:
	:math::
	`L_i = \\prod_{m = 0, m \\notin i}^{N - 1}\\frac{(x - x_m)}{(x_i - x_m)}`
	Parameters
	----------
	x : numpy.array
		A numpy array consisting of the :math: `x` nodes using which the
		lagrange basis functions need to be evaluated.
	Returns
	-------
	lagrange_basis_poly : numpy.ndarray
						  A :math: `N \\times N` matrix containing the
						  coefficients of the Lagrange basis polynomials such
						  that :math:`i^{th}` lagrange polynomial will be the
						  :math:`i^{th}` row of the matrix.
	'''
	xi_LGL = np.array(xi_LGL)
	lagrange_basis_poly = np.zeros([xi_LGL.shape[0], xi_LGL.shape[0]])
	
	for j in np.arange(xi_LGL.shape[0]):
		lagrange_basis_j = np.poly1d([1])
		
		for m in np.arange(xi_LGL.shape[0]):
			if m != j:
				lagrange_basis_j *= np.poly1d([1, -xi_LGL[m]]) \
									/ (xi_LGL[j] - xi_LGL[m])
		lagrange_basis_poly[j] = lagrange_basis_j.c
	
	return lagrange_basis_poly


def lagrange_basis_function():
	'''
	Funtion which calculates the value of lagrange basis functions over LGL
	nodes.
	
	Returns
	-------
	L_i    : arrayfire.Array [N N 1 1]
			 The value of lagrange basis functions calculated over the LGL
			 nodes.
	'''
	xi_tile     = af.transpose(af.tile(gvar.xi_LGL, 1, gvar.N_LGL))
	power       = af.flip(af.range(gvar.N_LGL))
	power_tile  = af.tile(power, 1, gvar.N_LGL)
	xi_pow      = af.arith.pow(xi_tile, power_tile)
	lBasisArray = af.interop.np_to_af_array(lagrange_basis_coeffs(gvar.xi_LGL))
	L_i         = af.blas.matmul(lBasisArray, xi_pow)
	
	return L_i

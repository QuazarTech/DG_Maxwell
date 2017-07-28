#! /usr/bin/env python3
import numpy as np
import arrayfire as af

from utils import utils

from app import global_variables as gvar

def LGL_points(N):
	"""
	Returns the :math: `N` Legendre-Gauss-Laguere points which are
	the roots of the equation
	:math::
		(1 - x^{2})L'_{N} = 0
	
	Parameters
	----------
	N : int
		Number of LGL-points to be generated.
		2 < N < 16
	
	Returns
	-------
	lgl : arrayfire.Array
		  An array of :math: `N` LGL points.
	"""
	if N > 16 or N < 2:
		print('Skipping! This function can only return from ',
			  '2 to 16 LGL points.')
	
	
	n = N - 2

	lgl = af.Array(gvar.LGL_list[n])
	return lgl


def lagrange_basis_coeffs(X):    
	'''
	A function to get the coefficients of the Lagrange
	basis polynomials for a given set of values.

	This function calculates the Lagrange basis
	polynomials by this formula:
	:math::
		L_i = \prod_{m = 0, m \notin i}^{N - 1}
				\frac{(x - x_m)}{(x_i - x_m)}

	Parameters
	----------
	X : numpy array
		The :math: `x` coordinates.

	Returns
	-------
	numpy.ndarray
	A :math: `N` x :math: `N` matrix containing the coefficients of the
	Lagrange basis polynomials such that :math:`i^{th}` lagrange polynomial 
	will be the :math:`i^{th}` row of the matrix.
	'''
	X = np.array(X)
	lagrange_basis_poly = np.zeros([X.shape[0], X.shape[0]])
	
	for j in np.arange(X.shape[0]):
		lagrange_basis_j = np.poly1d([1])
		
		for m in np.arange(X.shape[0]):
			if m != j:
				lagrange_basis_j *= np.poly1d([1, -X[m]]) \
									/ (X[j] - X[m])
		lagrange_basis_poly[j] = lagrange_basis_j.c
	
	return lagrange_basis_poly


def lagrange_basis(i, x):
	'''
	Calculates the value of the :math: `i^{th}` Lagrange basis (calculated
	using the gvar.xi_LGL points) at the :math: `x` coordinates.
	
	Parameters
	----------
	
	i : int
		The Lagrange basis which is to be calculated.
	
	x : af.Array
		The coordinates at which the `i^{th}` Lagrange polynomial is to be
		calculated.
	
	Returns
	-------
	
	lagrange : af.Array
			   Array of values of the :math: `i^{th}` Lagrange basis
			   calculated at the given :math: `x` coordinates.
	'''
	x_tile      = af.transpose(af.tile(x, 1, gvar.N_LGL))
	power       = utils.linspace((gvar.N_LGL-1), 0, gvar.N_LGL)
	power_tile  = af.tile(power, 1, x.shape[0])
	x_pow       = af.arith.pow(x_tile, power_tile)
	l_xi_j      = af.blas.matmul(gvar.lBasisArray[i], x_pow)
	
	return (l_xi_j)


def d_Lp_xi(element_nodes):
	'''
	Function which returns the value of the
	:math: `\\frac{d L_p(x_nodes)}{d\\xi}`
	as a 2D array of :math: `L_i x_{nodes}`.
	Where i varies along rows and the nodes vary along the columns.
	[TODO] : Complete.
	'''
	differentiation_coeffs = (af.transpose(af.flip(af.tile\
		(af.range(gvar.N_LGL), 1, gvar.N_LGL))) * gvar.lBasisArray)[:, :-1]
	
	nodes_tile         = af.transpose(af.tile(element_nodes, 1, gvar.N_LGL - 1))
	power_tile         = af.flip(af.tile\
									(af.range(gvar.N_LGL - 1), 1, gvar.N_LGL))
	nodes_power_tile   = af.pow(nodes_tile, power_tile)
	
	
	d_Lp_xi = af.blas.matmul(differentiation_coeffs, nodes_power_tile)
	
	return d_Lp_xi

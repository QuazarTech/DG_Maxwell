#! /usr/bin/env python3

import numpy as np
import arrayfire as af
from utils import utils
from scipy import special as sp

import global_variables as gvar

af.set_backend('cuda')
af.info()

def LGL_points(N):
	"""
	Takes N as an interger input and returns N LGL points.
	"""
	
	if N > 16:
		print('Skipping! This function can only return maximum 16 LGL points.')
		pass
	
	n = N - 2

	lgl = af.Array(gvar.LGL_list[n])
	return lgl

def lagrange_basis(X, i, x):
    '''
    X = X nodes
    i = i_{th} Lagrange Basis
    x = coordinate at which the i_{th}
        Lagrange polynomials are to be evaluated.
    '''
    
    for m in range(X.shape[0]):
        if m != i:
            lx *= (x-X[m])/(X[i]-X[m])
    
    return lx

if __name__ == '__main__':
	number_of_elements = 2

	X = utils.linspace(-1, 1, number_of_elements + 1)

	N = 16
	xi_LGL = LGL_points(N)
	af.display(xi_LGL)

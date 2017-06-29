#! /usr/bin/env python3

import numpy as np
import arrayfire as af
from utils import utils
from scipy import special as sp

import app.global_variables as gvar

af.set_backend('cuda')
af.info()

def LGL_points(N):
	"""
	Takes N as an interger input and returns N LGL points.
	"""
	
	if N > 16 or N < 2:
		print('Skipping! This function can only return from 2 to 16 LGL points.')
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
    lx = 1.
    
    for m in range(X.shape[0]):
        if m != i:
            lx *= (x - X[m]) / (X[i] - X[m])
    
    return lx

if __name__ == '__main__':
	number_of_elements = 15
	
	x = utils.linspace(-1, 1, 100)
	af.display(X)
	
	xi_LGL = LGL_points(number_of_elements + 1)
	af.display(xi_LGL)
	
	i = 9
	
	print(lagrange_basis(xi_LGL, i, x[50]))
	
	pass


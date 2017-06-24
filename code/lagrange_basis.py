#! /usr/bin/env python3

import numpy as np
import arrayfire as af
from matplotlib import pyplot as plt

af.set_backend('cuda')
af.info()

def lagrange_basis(X, i, x):
    '''
    X = X nodes
    i = i_{th} Lagrange Basis
    x = coordinate at which the i_{th}
        Lagrange polynomials are to be evaluated.
    '''
    
    lx = 0.
    
    for m in range(X.shape[0]):
        if m != i:
            lx *= (x-X[m])/(X[i]-X[m])
    
    return lx

def add(a, b):
	'''
	'''
	return a + b

def divide(a, b):
	'''
	'''
	return a / b

def multiply(a, b):
	'''
	'''
	return a * b

def linspace(start, end, number_of_points):
	'''
	Linspace implementation using arrayfire.
	'''
	X = af.range(number_of_points)
	d = (end - start) / (number_of_points - 1)
	X = af.broadcast(multiply, X, d)
	X = af.broadcast(add, X, start)
	
	return X

number_of_elements = 9

X = linspace(-5, 5, number_of_elements + 1)
Y = np.linspace(-5, 5, number_of_elements + 1)

af.display(X)


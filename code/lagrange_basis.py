#! /usr/bin/env python3

import numpy as np
import arrayfire as af
from matplotlib import pyplot as plt
from utils import utils

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

number_of_elements = 9

X = utils.linspace(-5, 5, number_of_elements + 1)

af.display(X)


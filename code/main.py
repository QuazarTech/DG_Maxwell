#! /usr/bin/env python3

import arrayfire as af
af.set_backend('opencl')
import math
import numpy as np

from matplotlib import pyplot as plt
from app import lagrange
from utils import utils
from app import global_variables as gvar
from unit_test import test_waveEqn
#from unit_test import lagrange_polynomial


if __name__ == '__main__':
	'''
	'''
    
	gvar.populateGlobalVariables()
	
	lagrange_polynomial.lagrangePolynomialTest()
	
	pass

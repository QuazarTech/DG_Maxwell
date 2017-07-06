#! /usr/bin/env python3

import arrayfire as af
import math
import numpy as np

from matplotlib import pyplot as plt
from app import lagrange
from utils import utils
from app import global_variables as gvar

from unit_tests import lagrange_polynomial


if __name__ == '__main__':
	'''
	'''
	af.set_backend('cuda')
	af.info()
	
	gvar.populateGlobalVariables()
	lagrange_polynomial.lagrangePolynomialTest()
	
	pass

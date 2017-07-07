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

from app import wave_equation as wave_eq
#from unit_test import lagrange_polynomial


if __name__ == '__main__':
	'''
	'''
	gvar.populateGlobalVariables()
	
	dx_dxi_array = wave_eq.dx_dxi(af.transpose(gvar.x_nodes[0]), gvar.xi_LGL)
	print(dx_dxi_array)
	
	pass

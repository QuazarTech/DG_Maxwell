#! /usr/bin/env python3

import arrayfire as af
af.set_backend('opencl')
import numpy as np
from app import global_variables as gvar
from unit_test import test_waveEqn
from app import wave_equation
from app import lagrange
if __name__ == '__main__':
	'''
	Main Function which sets backend and pouplates the global global_variables.
	Can be used to obtain results from other modules.
	'''
	gvar.populateGlobalVariables(8)
	element1_x_nodes = gvar.element_nodes[0 : 1]
	

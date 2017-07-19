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
	gvar.populateGlobalVariables(8, 12)
	#wave_equation.A_matrix()
	print(wave_equation.volume_integral_flux(gvar.u[0]))

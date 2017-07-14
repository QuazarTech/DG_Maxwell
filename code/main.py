#! /usr/bin/env python3

import arrayfire as af
af.set_backend('opencl')
from unit_test import test_waveEqn
from app import global_variables as gvar
from app import wave_equation

if __name__ == '__main__':
	'''
	Main Function which sets backend and pouplates the global global_variables.
	Can be used to obtain results from other modules.
	'''
	gvar.populateGlobalVariables(8)
	print(wave_equation.A_matrix())

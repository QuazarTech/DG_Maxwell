#! /usr/bin/env python3

import arrayfire as af
af.set_backend('opencl')
#from matplotlib import pyplot as plt
from app import global_variables as gvar

from app import wave_equation as wave_eq


if __name__ == '__main__':
	'''
	Main Function which sets backend and pouplates the global global_variables.
	Can be used to obtain results from other modules.
	'''
	gvar.populateGlobalVariables()
	

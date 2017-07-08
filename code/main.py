#! /usr/bin/env python3

import arrayfire as af
from matplotlib import pyplot as plt
from app import global_variables as gvar

from app import wave_equation as wave_eq


if __name__ == '__main__':
	'''
	'''
	af.set_backend('cuda')
	gvar.populateGlobalVariables()
	
	pass

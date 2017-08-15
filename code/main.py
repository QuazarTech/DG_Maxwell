#! /usr/bin/env python3

from os import sys

import arrayfire as af
af.set_backend(sys.argv[1])

from app import global_variables as gvar
from app import wave_equation


if __name__ == '__main__':
	
	gvar.populateGlobalVariables(8)
	print(wave_equation.time_evolution())

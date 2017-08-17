#! /usr/bin/env python3

from os import sys

import arrayfire as af
af.set_backend('cuda')

from app import global_variables as gvar
from app import wave_equation


if __name__ == '__main__':
	
	gvar.populateGlobalVariables(9)
	wave_equation.time_evolution()

#! /usr/bin/env python3

from os import sys

import arrayfire as af
af.set_backend('opencl')
from app import global_variables as gvar

if __name__ == '__main__':
	
	gvar.populateGlobalVariables(8)

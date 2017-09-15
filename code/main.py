#! /usr/bin/env python3

import arrayfire as af
import numpy as np
af.set_backend('opencl')

from app import params
from unit_test import test_waveEqn
from app import wave_equation
from app import lagrange


if __name__ == '__main__':
    
    wave_equation.time_evolution()

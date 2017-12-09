#! /usr/bin/env python3

import arrayfire as af
af.set_backend('opencl')
af.set_device(1)
af.info()

from dg_maxwell import wave_equation
from dg_maxwell import lagrange

if __name__ == '__main__':
    u_diff = wave_equation.time_evolution()
    print(lagrange.L1_norm(u_diff))

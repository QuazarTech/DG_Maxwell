#! /usr/bin/env python3

from dg_maxwell import wave_equation
from dg_maxwell import lagrange


if __name__ == '__main__':
    u_diff = wave_equation.time_evolution()
    print(lagrange.L1_norm(u_diff))
    wave_equation.convergence_test()

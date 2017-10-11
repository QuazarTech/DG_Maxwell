from dg_maxwell import params
from dg_maxwell import wave_equation
from dg_maxwell import lagrange
from dg_maxwell import utils
import arrayfire as af
import numpy as np


if __name__ == '__main__':
    print(wave_equation.time_evolution())

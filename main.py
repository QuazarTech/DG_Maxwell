import arrayfire as af
import numpy as np
import os
import numpy as np
import arrayfire as af

from matplotlib import pyplot as pl
from tqdm import trange

from dg_maxwell import wave_equation
from dg_maxwell import msh_parser
from dg_maxwell import wave_equation_2d
from dg_maxwell.tests import test_waveEqn
from dg_maxwell import isoparam
from dg_maxwell import lagrange
from dg_maxwell import params
from dg_maxwell import advection_2d
from dg_maxwell import utils

af.set_backend(params.backend)

print(af.info())
print(advection_2d.time_evolution())

<<<<<<< HEAD
import arrayfire as af
import numpy as np
import os
=======
import numpy as np
import arrayfire as af

>>>>>>> 24ad7811eefdc05cb59c8676ce8659685e9a8599
from matplotlib import pyplot as pl

from dg_maxwell import params
from dg_maxwell import lagrange
<<<<<<< HEAD
from dg_maxwell import wave_equation_2d
from dg_maxwell import wave_equation
from dg_maxwell import utils
from dg_maxwell import isoparam
from dg_maxwell import msh_parser

poly1_coeffs = af.reorder(af.transpose(af.np_to_af_array(np.array([[1, 2, 3.], [1, 2, 3]]))), 0, 2, 1)
poly2_coeffs = af.reorder(af.transpose(af.np_to_af_array(np.array([[-2, 4, 7.], [-2, 4, 7.]]))), 0, 2, 1)
print(poly1_coeffs, poly2_coeffs)
print(utils.polynomial_product_coeffs(poly1_coeffs, poly2_coeffs))
print(wave_equation_2d.Li_Lj_coeffs()[:, :, 1])
=======
from dg_maxwell import wave_equation
from dg_maxwell import msh_parser
from dg_maxwell import wave_equation_2d
from dg_maxwell.tests import test_waveEqn
from dg_maxwell import isoparam
from dg_maxwell import utils

af.set_backend(params.backend)



>>>>>>> 24ad7811eefdc05cb59c8676ce8659685e9a8599

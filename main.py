import numpy as np
import arrayfire as af


from dg_maxwell import params
from dg_maxwell import lagrange
from dg_maxwell import wave_equation
from dg_maxwell import wave_equation_2d
from dg_maxwell import isoparam
from dg_maxwell import utils

af.set_backend(params.backend)

diff_l0 = params.dl_dxi_coeffs[0]
xi = params.xi_LGL
u_i = np.e**(-(xi)**2/0.6**2)
#print(u_i)
F_ij = af.sum(af.broadcast(utils.multiply, u_i, params.lagrange_coeffs), 0)
#print(F_ij)
vol_int_coeffs = af.convolve1(af.transpose(diff_l0), af.transpose(F_ij), conv_mode=af.CONV_MODE.EXPAND)
#print(vol_int_coeffs)
af.display(lagrange.integrate(F_ij), 14)
#int_coeffs = (af.sum(af.tile(F_ij, 1, params.N_LGL) * params.lagrange_coeffs, 0))
#af.display(lagrange.integrate(int_coeffs), 14)
#num_vol_int = (wave_equation.volume_integral_flux(params.u_init))

reference_vol_int = af.transpose(af.interop.np_to_af_array(np.array
        ([
        [-0.002016634876668093, -0.000588597708116113, -0.0013016773719126333,\
        -0.002368387579324652, -0.003620502047659841, -0.004320197094090966,
        -0.003445512010153811, 0.0176615086879261],\

        [-0.018969769374, -0.00431252844519,-0.00882630935977,-0.0144355176966,\
        -0.019612124119, -0.0209837936827, -0.0154359890788, 0.102576031756], \

        [-0.108222418798, -0.0179274222595, -0.0337807018822, -0.0492589052599,\
        -0.0588472807471, -0.0557970236273, -0.0374764132459, 0.361310165819],\

        [-0.374448714304, -0.0399576371245, -0.0683852285846, -0.0869229749357,\
        -0.0884322503841, -0.0714664112839, -0.0422339853622, 0.771847201979], \

        [-0.785754362849, -0.0396035640187, -0.0579313769517, -0.0569022801117,\
        -0.0392041960688, -0.0172295769141, -0.00337464521455, 1.00000000213],\

        [-1.00000000213, 0.00337464521455, 0.0172295769141, 0.0392041960688,\
        0.0569022801117, 0.0579313769517, 0.0396035640187, 0.785754362849],\

        [-0.771847201979, 0.0422339853622, 0.0714664112839, 0.0884322503841, \
        0.0869229749357, 0.0683852285846, 0.0399576371245, 0.374448714304],\

        [-0.361310165819, 0.0374764132459, 0.0557970236273, 0.0588472807471,\
        0.0492589052599, 0.0337807018822, 0.0179274222595, 0.108222418798], \

        [-0.102576031756, 0.0154359890788, 0.0209837936827, 0.019612124119, \
        0.0144355176966, 0.00882630935977, 0.00431252844519, 0.018969769374],\

        [-0.0176615086879, 0.00344551201015 ,0.00432019709409, 0.00362050204766,\
        0.00236838757932, 0.00130167737191, 0.000588597708116, 0.00201663487667]\
        ])))
#print(af.max(af.abs(num_vol_int - reference_vol_int)))

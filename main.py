from dg_maxwell import params
import arrayfire as af
import numpy as np
from dg_maxwell import wave_equation
from dg_maxwell import lagrange
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'  ] = 9.6, 6.
plt.rcParams['figure.dpi'      ] = 100
plt.rcParams['image.cmap'      ] = 'jet'
plt.rcParams['lines.linewidth' ] = 1.5
plt.rcParams['font.family'     ] = 'serif'
plt.rcParams['font.weight'     ] = 'bold'
plt.rcParams['font.size'       ] = 20
plt.rcParams['font.sans-serif' ] = 'serif'
plt.rcParams['text.usetex'     ] = True
plt.rcParams['axes.linewidth'  ] = 1.5
plt.rcParams['axes.titlesize'  ] = 'medium'
plt.rcParams['axes.labelsize'  ] = 'medium'
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['xtick.major.pad' ] = 8
plt.rcParams['xtick.minor.pad' ] = 8
plt.rcParams['xtick.color'     ] = 'k'
plt.rcParams['xtick.labelsize' ] = 'medium'
plt.rcParams['xtick.direction' ] = 'in'
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['ytick.major.pad' ] = 8
plt.rcParams['ytick.minor.pad' ] = 8
plt.rcParams['ytick.color'     ] = 'k'
plt.rcParams['ytick.labelsize' ] = 'medium'
plt.rcParams['ytick.direction' ] = 'in'


if __name__ == '__main__':

    # REMEMBER TO DELETE THE END OF THE FILE

    #print(lagrange.lagrange_polynomials(params.xi_LGL))
#    poly1 = af.Array([0, 0, 1, 0, 1, 0, 0])
#    poly1 = af.tile(poly1, 1, 2)
#    poly2 = af.Array([0, 0, 2, 7, 0, 0])
#    poly2 = af.tile(poly2, 1, 2)
#    poly2 = af.reorder(poly2, 0, 2, 1)
#    
#    #(af.np_to_af_array(np.array([[0., 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0], \
#    #                                   [0., 0, 0, 0, 0, 0, 0, 2, 3, 7,-2, 2, 8, 2, 0, 0, 0, 0, 0, 0, 0]])))
#
#    print(poly1, poly2, af.convolve(poly1, poly2))

    test_poly1 = af.transpose(af.np_to_af_array(np.array([[0,0, 0, 0, 1, 0, 1, 3,0, 0, 0,0], [0,0, 0,0, 3, 5., -7, 9,0, 0,0, 0]])))
    test_poly2 = af.reorder(\
                 af.transpose(af.np_to_af_array(np.array([[0,0, 0, 0, 2, 7, -8, 3,1, 0,0, 0,0],\
                                   [0, 0,0, 0, 7,4, 3, 7, -6, 0, 0,0, 0], [0, 0, 0,0,2, 13, -2., 1, 7,0,0,0, 0]]))), 0, 2, 1)

#    print(af.convolve(test_poly1, test_poly2))
    #print(params.lagrange_coeffs)
#    print((lagrange.lagrange_interpolation_u(params.u_init)))

    print(wave_equation.volume_integral_flux(params.u_init, 0))
    print(wave_equation.time_evolution())
#
#    test_array = af.constant(9, 1, 7)
#    test_value = af.constant(2, 1, 3)
#    print(test_array[0, 2:5], test_value)
#    test_array[0, 2:5] = test_value
#    print(lagrange.lagrange_interpolation_u(params.u_init))

























#    #wave_equation.change_parameters(8, 10)
#    #print(wave_equation.convergence_test())
#    #print(params.differential_lagrange_polynomial)
#    test_nodes = (af.transpose(params.element_array[0]))
#   # print(wave_equation.analytical_volume_integral(test_nodes, 0))
#   # print(af.display(wave_equation.volume_integral_flux(params.u_init, 0)[0,0], 14))
#   #print(lagrange.lagrange_polynomials(params.xi_LGL)[0][1].deriv)
#    L1_norm_error = np.zeros([11])
#    N_LGL = (np.arange(11) + 3).astype(float)
#    for p in range(3, 14):
#        wave_equation.change_parameters(p, 10)
#        volume_integral = np.zeros([params.N_LGL, params.N_Elements])
#        for i in range(0, params.N_Elements):
#            test_nodes = af.transpose(params.element_array[i])
#            for j in range(0, params.N_LGL):
#                volume_integral[j][i] = (wave_equation.analytical_volume_integral(test_nodes, j))
#        volume_integral = (af.np_to_af_array(volume_integral))
#        L1_norm_error[p-3] = (af.mean(af.abs(volume_integral - wave_equation.volume_integral_flux(params.u_init,0))))
#
#    normalization = 0.00924 / (3 ** (-3))
#    print(L1_norm_error)
#
#    plt.loglog(N_LGL, L1_norm_error, marker='o', label='L1 norm')
#    plt.loglog(N_LGL, normalization * N_LGL **(-N_LGL), color='black', linestyle='--', label='$N_{LGL}^{-N_{LGL}}$')
#    plt.xlabel('LGL points')
#    plt.ylabel('L1 norm')
#    plt.legend(loc='best')
#    plt.show()

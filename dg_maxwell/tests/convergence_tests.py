import arrayfire as af
import numpy as np

from dg_maxwell import params
from dg_maxwell import wave_equation

af.set_device(params.device)
af.set_backend(params.backend)

def L1_norm(u):
    '''

    A function to calculate the L1 norm of error using
    the polynomial obtained using Lagrange interpolation

    Parameters
    ----------
    u : arrayfire.Array [N_LGL N_Elements 1 1]
        Difference between analytical and numerical u at the mapped LGL points.

    Returns
    -------
    L1_norm : float64
              The L1 norm of error.

    '''
    interpolated_coeffs = af.reorder(lagrange.lagrange_interpolation_u(\
                                           u), 2, 1, 0)

    L1_norm = af.sum(lagrange.integrate(interpolated_coeffs))

    return L1_norm

def convergence_test():
    '''

    Used to obtain plots of L1 norm versus parameters (Number of elements
    or N_LGL).
    
    '''
    L1_norm_option_1 = np.zeros([15])
    N_lgl            = (np.arange(15) + 3).astype(float)
    L1_norm_option_3 = np.zeros([15])

    for i in range(0, 15):
        test_waveEqn.change_parameters(i + 3, 15, i + 3)
        u_diff = wave_equation.time_evolution()
        L1_norm_option_1[i] = L1_norm(u_diff)
        test_waveEqn.change_parameters(i + 3, 15, i + 4)
        u_diff = wave_equation.time_evolution()
        L1_norm_option_3[i] = L1_norm(u_diff)


    print(L1_norm_option_1, L1_norm_option_3)
    normalization = 0.00281 / (3 **(-3))
    plt.loglog(N_lgl, L1_norm_option_1, marker='o', label='option 1')
    plt.loglog(N_lgl, L1_norm_option_3, marker='o', label='option 3')
    plt.xlabel('No. of LGL points')
    plt.ylabel('L1 norm of error')
    plt.title('L1 norm after 1 full advection')
    plt.loglog(N_lgl, normalization * N_lgl **(-N_lgl), color='black',\
                          linestyle='--', label='$N_{LGL}^{-N_{LGL}}$')
    plt.legend(loc='best')

    plt.show()

    return

def analytical_volume_integral(x_nodes, p):
    '''
    Computes the volume integral analytically for a sin(2 * pi * x) wave from
    -1 to 1.
    '''
    dlp_dxi = params.differential_lagrange_polynomial[p]

    def F_u(x):
        analytical_flux = wave_equation.flux_x(np.sin(2* np.pi*((x_nodes[1] - x_nodes[0])\
                       * x/2 + (x_nodes[0] + x_nodes[1]) / 2))) * dlp_dxi(x)
        return analytical_flux

    analytical_integral = integrate.quad(F_u, -1, 1)[0]


    return analytical_integral


def volume_int_convergence():
    '''
    convergence test for volume int flux
    '''
    N_LGL = np.arange(15).astype(float) + 3
    L1_norm_option_3 = np.zeros([15])
    L1_norm_option_1 = np.zeros([15])
    for i in range(0, 15):
        test_waveEqn.change_parameters(i + 3, 16, i + 4)
        vol_int_analytical = np.zeros([params.N_Elements, params.N_LGL])
        for j in range (params.N_Elements):
            for k in range (params.N_LGL):
                vol_int_analytical[j][k] = (analytical_volume_integral\
                             (af.transpose(params.element_array[j]), k))
        vol_int_analytical = af.transpose(af.np_to_af_array\
                                                   (vol_int_analytical))
        L1_norm_option_3[i] = af.mean(af.abs(vol_int_analytical\
                                      - wave_equation.volume_integral_flux(params.u_init, 0)))


    for i in range(0, 15):
        test_waveEqn.change_parameters(i + 3, 16, i + 3)
        vol_int_analytical = np.zeros([params.N_Elements, params.N_LGL])
        for j in range (params.N_Elements):
            for k in range (params.N_LGL):
                vol_int_analytical[j][k] = analytical_volume_integral(\
                                           af.transpose(params.element_array[j]), k)
        vol_int_analytical  = af.transpose(af.np_to_af_array(vol_int_analytical))
        L1_norm_option_1[i] = af.mean(af.abs(vol_int_analytical\
                                      - wave_equation.volume_integral_flux(params.u_init, 0)))
    normalization = 0.0023187 / (3 ** (-3))


    print(L1_norm_option_1, L1_norm_option_3)
    plt.loglog(N_LGL, L1_norm_option_1, marker='o', label='L1 norm option 1')
    plt.loglog(N_LGL, L1_norm_option_3, marker='o', label='L1 norm option 3')
    plt.loglog(N_LGL, normalization * N_LGL **(-N_LGL), color='black', linestyle='--', label='$N_{LGL}^{-N_{LGL}}$')
    plt.title('L1 norm of volume integral term')
    plt.xlabel('LGL points')
    plt.ylabel('L1 norm')
    plt.legend(loc='best')
    plt.show()

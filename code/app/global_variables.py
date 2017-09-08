import numpy as np
import arrayfire as af
af.set_backend('opencl')
from scipy import special as sp
from app import lagrange
from app import wave_equation
from utils import utils
import math


# This module is used to change the parameters of the simulation.

# The functions to calculate Legendre-Gauss-Lobatto points and
# the Lobatto weights used for integration are given below.

def LGL_points(N):
    '''
    Calculates and returns the LGL points for a given N, These are roots of
    the formula.
    :math: `(1 - xi ** 2) P_{n - 1}'(xi) = 0`
    Legendre polynomials satisfy the recurrence relation
    :math:`(1 - x ** 2) P_n' (x) = -n x P_n(x) + n P_{n - 1} (x)`

    Parameters
    ----------
    N : Int
        Number of LGL nodes required
    
    Returns
    -------
    lgl : arrayfire.Array [N 1 1 1]
          An arrayfire array consisting of the Lagrange-Gauss-Lobatto Nodes.
                          
    Reference
    ---------
    http://mathworld.wolfram.com/LobattoQuadrature.html
    '''
    xi                 = np.poly1d([1, 0])
    legendre_N_minus_1 = N * (xi * sp.legendre(N - 1) - sp.legendre(N))
    lgl_points         = legendre_N_minus_1.r
    lgl_points.sort()
    lgl_points         = af.np_to_af_array(lgl_points)

    return lgl_points

def lobatto_weight_function(n, x):
    '''
    Calculates and returns the weight function for an index :math:`n`
    and points :math: `x`.
    
    :math::
        `w_{n} = \\frac{2 P(x)^2}{n (n - 1)}`,
        Where P(x) is $ (n - 1)^th $ index.
    
    Parameters
    ----------
    n : int
        Index for which lobatto weight function
    
    x : arrayfire.Array [N 1 1 1]
        1D array of points where weight function is to be calculated.
    
    
    Returns
    -------

    gauss_lobatto_weights : arrayfire.Array
                            An array of lobatto weight functions for
                            the given :math: `x` points and index.
    Reference
    ---------
    Gauss-Lobatto weights Wikipedia link-
    https://en.wikipedia.org/wiki/
    Gaussian_quadrature#Gauss.E2.80.93Lobatto_rules
    
    '''
    P = sp.legendre(n - 1)
    
    gauss_lobatto_weights = (2 / (n * (n - 1)) / (P(x))**2)
    
    return gauss_lobatto_weights




# The domain of the function which is of interest.
x_nodes    = af.np_to_af_array(np.array([-1., 1.]))

# The number of LGL points into which an element is split
N_LGL      = 8 

# Number of elements the domain is to be divided into
N_Elements = 10 

# The speed of the wave.
c          = 1 

# The total time for which the simulation is to be carried out.
total_time = 1 

# The c_lax to be used in the Lax-Friedrichs flux.
c_lax      = 1

# Array containing the LGL points in xi space.
xi_LGL     = LGL_points(N_LGL)


# Array of the lobatto weights used during integration.
lobatto_weights = af.np_to_af_array(lobatto_weight_function(N_LGL, xi_LGL)) 
 
# An array containing the coefficients of the lagrange basis polynomials.
lagrange_coeffs = af.np_to_af_array(lagrange.lagrange_basis_coeffs(xi_LGL))

# Refer corresponding functions.
dLp_xi               = lagrange.dLp_xi_LGL(lagrange_coeffs)
lagrange_basis_value = lagrange.lagrange_basis_function(lagrange_coeffs) 

# Obtaining an array consisting of the LGL points mapped onto the elements.
element_size    = af.sum((x_nodes[1] - x_nodes[0]) / N_Elements)
elements_xi_LGL = af.constant(0, N_Elements, N_LGL)
elements        = utils.linspace(af.sum(x_nodes[0]),
                                      af.sum(x_nodes[1] - element_size),
                                      N_Elements)
    
np_element_array   = np.concatenate((af.transpose(elements), 
                           af.transpose(elements + element_size)))
    
element_mesh_nodes = utils.linspace(af.sum(x_nodes[0]),
                                      af.sum(x_nodes[1]),
                                      N_Elements + 1)
    
element_array = af.transpose(af.interop.np_to_af_array(np_element_array))
element_LGL   = wave_equation.mapping_xi_to_x(af.transpose(element_array), xi_LGL)


# The minimum distance between 2 mapped LGL points.
delta_x = af.min((element_LGL - af.shift(element_LGL, 1, 0))[1:, :])

# The value of time-step.
delta_t = delta_x / (20 * c)

# Array of timesteps seperated by delta_t.
time    = utils.linspace(0, int(total_time / delta_t) * delta_t,
                                            int(total_time / delta_t))

# Initializing the amplitudes. Change u_init to required initial conditions.
u_init     = np.e ** (-(element_LGL) ** 2 / 0.4 ** 2)
u          = af.constant(0, N_LGL, N_Elements, time.shape[0],\
                         dtype = af.Dtype.f64)
u[:, :, 0] = u_init

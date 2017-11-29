#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import csv

import numpy as np
import matplotlib.lines as lines
import arrayfire as af

from dg_maxwell import params
from dg_maxwell import lagrange

af.set_backend(params.backend)
af.set_device(params.device)

def add(a, b):
    '''
    For broadcasting purposes, To sum two arrays of different
    shapes, A function which can sum two variables is required.

    Parameters
    ----------
    a : arrayfire.Array [N M 1 1]
        One of the arrays which need to be broadcasted and summed.

    b : arrayfire.Array [1 M L 1]
        One of the arrays which need to be broadcasted and summed.

    Returns
    -------
    add : arrayfire.Array [N M L 1]
          Sum of a and b. When used along with ``af.broadcast``
          can be used to sum different size arrays.
    '''
    add = a + b

    return add


def divide(a, b):
    '''

    For broadcasting purposes, To divide two arrays of different
    shapes, A function which can sum two variables is required.

    Parameters
    ----------
    a : arrayfire.Array [N M 1 1]
        One of the arrays which need to be broadcasted and divided.

    b : arrayfire.Array [1 M L 1]
        One of the arrays which need to be broadcasted and divided.

    Returns
    -------
    quotient : arrayfire.Array [N M L 1]
               The quotient a / b. When used along with af.broadcast
               can be used to give quotient of two different size arrays
               by dividing elements of the broadcasted array.

    '''
    quotient = a / b

    return quotient


def multiply(a, b):
    '''
    For broadcasting purposes, To divide two arrays of different
    shapes, A function which can sum two variables is required.

    Parameters
    ----------
    a : arrayfire.Array [N M 1 1]
        One of the arrays which need to be broadcasted and multiplying.

    b : arrayfire.Array [1 M L 1]
        One of the arrays which need to be broadcasted and multiplying.

    Returns
    -------
    product : arrayfire.Array [N M L 1]
              The product a * b . When used along with af.broadcast
              can be used to give quotient of two different size arrays
              by multiplying elements of the broadcasted array.
    '''

    product = a * b

    return product

def power(a, b):
    '''
    For broadcasting purposes, To divide two arrays of different
    shapes, A function which can sum two variables is required.

    Parameters
    ----------
    a : arrayfire.Array [N M 1 1]
        One of the arrays which need to be broadcasted and multiplying.

    b : arrayfire.Array [1 M L 1]
        One of the arrays which need to be broadcasted and multiplying.

    Returns
    -------
    power : arrayfire.Array [N M L 1]
            The quotient a / b. When used along with af.broadcast
            can be used to give quotient of two different size arrays
            by multiplying elements of the broadcasted array.
    '''
    power  = a ** b

    return power



def linspace(start, end, number_of_points):
    '''
    Linspace implementation using arrayfire.

    Returns
    -------
    X : arrayfire.Array
        An array which contains ``number_of_points`` evenly spaced points
        between ``start`` and ``end``.
    '''
    X = af.range(number_of_points, dtype = af.Dtype.f64)
    d = (end - start) / (number_of_points - 1)
    X = X * d
    X = X + start

    return X

def plot_line(points, axes_handler, grid_width = 2., grid_color = 'blue'):
    '''
    Plots curves using the given :math:`(x, y)` points. It joins the
    points using lines in the given order.

    Parameters
    ----------
    points       : np.ndarray [N, 2]
                   :math:`(x, y)` coordinates of :math:`N` points. First and
                   second column stores :math:`x` and :math:`y` coordinates
                   of an point.

    axes_handler : matplotlib.axes.Axes
                   The plot handler being used to plot the element grid.
                   You may generate it by calling the function pyplot.axes()

    grid_width   : float
                   Grid line width.

    grid_color   : str
                   Grid line color.

    Returns
    -------

    None
    '''

    for point_id in np.arange(1, len(points)):
        line = [points[point_id].tolist(), points[point_id - 1].tolist()]
        (line1_xs, line1_ys) = zip(*line)
        axes_handler.add_line(lines.Line2D(line1_xs, line1_ys,
                                           linewidth = grid_width,
                                           color = grid_color))

    return

def csv_to_numpy(filename, delimeter_ = ','):
    '''
    Reads a text file data and converts it into a numpy :math:`2D` numpy
    array.

    Parameters
    ----------
    filename : str
               File which is to be read.

    delimeter : str
                Delimeter used in the document.

    Returns
    -------
    content : np.array
              Read content from the file.
    '''

    csv_handler = csv.reader(open(filename, newline='\n'),
                             delimiter = delimeter_)

    content = list()

    for line in csv_handler:
        content.append(list())
        for item in line:
            try:
                content[-1].append(float(item))
            except ValueError:
                if content[-1] == []:
                    content.pop()
                    print('popping string')
                break

    content = np.array(content, dtype = np.float64)

    return content

def af_meshgrid(arr_0, arr_1):
    '''
    Creates a meshgrid from the given two arrayfire array.

    Parameters
    ----------

    arr_0 : af.Array [N_0 1 1 1]

    arr_1 : af.Array [N_1 1 1 1]

    Returns
    -------

    tuple(af.Array[N_1 N_0 1 1], af.Array[N_1 N_0 1 1])
    '''

    Arr_0 = af.tile(af.transpose(arr_0), d0 = arr_1.shape[0])
    Arr_1 = af.tile(arr_1, d0 = 1, d1 = arr_0.shape[0])

    return Arr_0, Arr_1


def outer_prod(a, b):
    '''
    Calculates the outer product of two matrices.

    Parameters
    ----------
    a : af.Array [N_a N 1 1]

    b : af.Array [N_b N 1 1]

    Returns
    -------

    af.Array [N_a N_b N 1]
    Outer product of two elements
    '''

    if id(a) == id(b):
        array_a = a.copy()
        array_b = b.copy()
    else:
        array_a = a
        array_b = b

    a_n1 = array_a.shape[0]
    b_n1 = array_b.shape[0]
    
    if (a.numdims() == 1) & (b.numdims() == 1):
        a_n2 = 1
        b_n2 = 1
    else:
        a_n2 = array_a.shape[1]
        b_n2 = array_b.shape[1]

    a_reorder = af.reorder(array_a, d0 = 0, d1 = 2, d2 = 1)
    b_reorder = af.reorder(array_b, d0 = 0, d1 = 2, d2 = 1)
    b_reorder = af.transpose(b_reorder)

    a_tile = af.tile(a_reorder, d0 = 1, d1 = b_n1)
    b_tile = af.tile(b_reorder, d0 = a_n1)

    return a_tile * b_tile


def matmul_3D(a, b):
    '''
    Finds the matrix multiplication of :math:`Q` pairs of matrices ``a`` and
    ``b``.

    Parameters
    ----------
    a : af.Array [M N Q 1]
        First set of :math:`Q` 2D arrays :math:`N \\neq 1` and :math:`M \\neq 1`.

    b : af.Array [N P Q 1]
        Second set of :math:`Q` 2D arrays :math:`P \\neq 1`.

    Returns
    -------
    matmul : af.Array [M P Q 1]
             Matrix multiplication of :math:`Q` sets of 2D arrays.
    '''
    shape_a = shape(a)
    shape_b = shape(b)

    P = shape_b[1]

    a = af.transpose(a)
    a = af.reorder(a, d0 = 0, d1 = 3, d2 = 2, d3 = 1)
    a = af.tile(a, d0 = 1, d1 = P)
    b = af.tile(b, d0 = 1, d1 = 1, d2 = 1, d3 = a.shape[3])

    matmul = af.sum(a * b, dim = 0)
    matmul = af.reorder(matmul, d0 = 3, d1 = 1, d2 = 2, d3 = 0)

    return matmul


def shape(array):
    '''
    '''
    af_shape = array.shape

    shape = [1, 1, 1, 1]

    for dim in np.arange(array.numdims()):
        shape[dim] = af_shape[dim]

    return shape


def polyval_1d(polynomials, xi):
    '''
    Finds the value of the polynomials at the given :math:`\\xi` coordinates.

    Parameters
    ----------
    polynomials : af.Array [number_of_polynomials N 1 1]
                 ``number_of_polynomials`` :math:`2D` polynomials of degree
                 :math:`N - 1` of the form

                 .. math:: P(x) = a_0x^0 + a_1x^1 + ... \\
                           a_{N - 1}x^{N - 1} + a_Nx^N

    xi      : af.Array [N 1 1 1]
              :math:`\\xi` coordinates at which the :math:`i^{th}` Lagrange
              basis polynomial is to be evaluated.

    Returns
    -------
    af.Array [i.shape[0] xi.shape[0] 1 1]
        Evaluated polynomials at given :math:`\\xi` coordinates
    '''

    N     = int(polynomials.shape[1])
    xi_   = af.tile(af.transpose(xi), d0 = N)
    power = af.tile(af.flip(af.range(N), dim = 0),
                    d0 = 1, d1 = xi.shape[0])

    xi_power = xi_**power

    return af.matmul(polynomials, xi_power)


def poly1d_product(poly_a, poly_b):
    '''
    Finds the product of two polynomials using the arrayfire convolve1
    function.

    Parameters
    ----------
    poly_a : af.Array[N degree_a 1 1]
             :math:`N` polynomials of degree :math:`degree`

    poly_b : af.Array[N degree_b 1 1]
             :math:`N` polynomials of degree :math:`degree_b`
    '''
    return af.transpose(af.convolve1(af.transpose(poly_a),
                                     af.transpose(poly_b),
                                     conv_mode = af.CONV_MODE.EXPAND))


def integrate_1d(polynomials, order, scheme = 'gauss'):
    '''
    Integrates single variables using the Gauss-Legendre or Gauss-Lobatto
    quadrature.

    Parameters
    ----------
    polynomials : af.Array [number_of_polynomials degree 1 1]
                  The polynomials to be integrated.

    order       : int
                  Order of the quadrature.

    scheme      : str
                  Possible options are

                  - ``gauss`` for using Gauss-Legendre quadrature
                  - ``lobatto`` for using Gauss-Lobatto quadrature

    Returns
    -------
    integral : af.Array [number_of_polynomials 1 1 1]
               The integral for the respective polynomials using the given
               quadrature scheme.
    '''
    integral = 0.0

    if scheme == 'gauss':

        N_g = order
        xi_gauss      = af.np_to_af_array(lagrange.gauss_nodes(N_g))
        gauss_weights = lagrange.gaussian_weights(N_g)

        polyval_gauss = polyval_1d(polynomials, xi_gauss)

        integral = af.sum(af.transpose(af.broadcast(multiply,
                                                    af.transpose(polyval_gauss),
                                                    gauss_weights)), dim = 1)

        return integral
        
    elif scheme == 'lobatto':
        N_l = order
        xi_lobatto      = lagrange.LGL_points(N_l)
        lobatto_weights = lagrange.lobatto_weights(N_l)

        polyval_lobatto = polyval_1d(polynomials, xi_lobatto)

        integral = af.sum(af.transpose(af.broadcast(multiply,
                                                    af.transpose(polyval_lobatto),
                                                    lobatto_weights)), dim = 1)

        return integral

    else:
        return -1.



def integrate_2d(poly_xi, poly_eta, order, scheme = 'gauss'):
    '''
    Integrates functions defined by polynomials of :math:`\\xi` and
    :math:`\\eta` using the Gauss-Legendre or Gauss-Lobatto quadrature.

    Parameters
    ----------
    poly_xi : af.Array [number_of_polynomials N 1 1]
              ``number_of_polynomials`` polynomials of :math:`\\xi` with
              degree :math:`N - 1` of the form

                 .. math:: P_0(\\xi) = a_0\\xi^0 + a_1\\xi^1 + ... \\
                           a_{N - 1}\\xi^{N - 1}

    poly_eta : af.Array [number_of_polynomials N 1 1]
               ``number_of_polynomials`` polynomials of :math:`\\eta` with
               degree :math:`N - 1` of the form

               .. math:: P_1(\\eta) = a_0\\eta^0 + a_1\\eta^1 + ... \\
                         a_{N - 1}\\eta^{N - 1}

    order   : int
              Order of the Gauss-Legendre or Gauss-Lobatto quadrature.

    scheme  : str
              Possible options are

              - ``gauss`` for using Gauss-Legendre quadrature
              - ``lobatto`` for using Gauss-Lobatto quadrature

    Returns
    -------
    integrate_poly_xi_eta : af.Array [number_of_polynomials 1 1 1]
                            Integral

                            .. math:: \\iint P_0(\\xi) P_1(\\eta) d\\xi d\\eta
    '''
    
    integrate_poly_xi     = integrate_1d(poly_xi, order, scheme)
    integrate_poly_eta    = integrate_1d(poly_eta, order, scheme)
    integrate_poly_xi_eta = integrate_poly_xi * integrate_poly_eta

    return integrate_poly_xi_eta


def polynomial_derivative(polynomial):
    '''
    '''
    derivtive_multiplier = af.tile(af.transpose(af.flip(
        af.range(polynomial.shape[1]))),
                                   d0 = polynomial.shape[0])
    
    return (polynomial * derivtive_multiplier)[:, : -1]



def polynomial_product_coeffs(poly1_coeffs, poly2_coeffs):
    '''
    '''
    poly1_coeffs_tile = af.transpose(af.tile(poly1_coeffs, 1, poly1_coeffs.shape[0]))
    poly2_coeffs_tile = af.tile(poly2_coeffs, 1, poly2_coeffs.shape[0])

    product_coeffs = poly1_coeffs_tile * poly2_coeffs_tile

    return product_coeffs


def polyval_2d(poly_2d, xi, eta):
    '''
    '''
    poly_2d_shape = poly_2d.shape
    poly_xy = af.tile(poly_2d, d0 = 1, d1 = 1, d2 = 1, d3 = xi.shape[0])
    poly_xy_shape = poly_xy.shape
    # print(poly_xy)

    xi_power = af.flip(af.range(poly_xy_shape[1], dtype = af.Dtype.u32))
    xi_power = af.tile(af.transpose(xi_power), d0 = poly_xy_shape[0])
    xi_power = af.tile(xi_power, d0 = 1, d1 = 1, d2 = xi.shape[0])

    eta_power = af.flip(af.range(poly_xy_shape[0], dtype = af.Dtype.u32))
    eta_power = af.tile(eta_power, d0 = 1, d1 = poly_xy_shape[1])
    eta_power = af.tile(eta_power, d0 = 1, d1 = 1, d2 = eta.shape[0])

    Xi = af.reorder(xi, d0 = 2, d1 = 1, d2 = 0)
    Xi = af.tile(Xi, d0 = poly_xy_shape[0], d1 = poly_xy_shape[1])
    Xi = af.pow(Xi, xi_power)
    Xi = af.reorder(Xi, d0 = 0, d1 = 1, d2 = 3, d3 = 2)
    # print(Xi)

    Eta = af.reorder(eta, d0 = 2, d1 = 1, d2 = 0)
    Eta = af.tile(Eta, d0 = poly_xy_shape[0], d1 = poly_xy_shape[1])
    Eta = af.pow(Eta, eta_power)
    Eta = af.reorder(Eta, d0 = 0, d1 = 1, d2 = 3, d3 = 2)
    # print(Eta)

    Xi_Eta = Xi * Eta

    poly_val = af.broadcast(multiply, poly_xy, Xi_Eta)
    poly_val = af.sum(af.sum(poly_val, dim = 1), dim = 0)
    poly_val = af.reorder(poly_val, d0 = 2, d1 = 3, d2 = 0, d3 = 1)

    return poly_val

def multivariable_poly_value(poly_2d, x, y):
    '''
    '''
    polynomial_coeffs = af.transpose(af.moddims(poly_2d, poly_2d.shape[0] ** 2, poly_2d.shape[2]))

    power_index     = af.flip(af.np_to_af_array(np.arange(poly_2d.shape[0])))
    x_power_indices = af.flat(af.transpose(af.tile(power_index, 1, poly_2d.shape[0])))
    y_power_indices = af.tile(power_index, poly_2d.shape[0])

    x_power = af.broadcast(power, af.transpose(x), x_power_indices)
    y_power = af.broadcast(power, af.transpose(y), y_power_indices)

    polynomial_value = af.matmul(polynomial_coeffs, x_power * y_power)


    return polynomial_value

def gauss_quad_multivar_poly(poly_xi_eta, N_quad, advec_var):
    '''
    '''
    shape_poly_2d = poly_xi_eta.shape
    xi_gauss      = advec_var.gauss_points

    Xi  = af.flat(af.transpose(af.tile(xi_gauss, 1, params.N_quad)))
    Eta = af.tile(xi_gauss, params.N_quad)

    w_i = advec_var.gauss_weights

    test_W_i = af.flat(af.transpose(af.tile(w_i, 1, params.N_quad)))
    test_W_j = af.tile(w_i, params.N_quad)

    W_i = af.tile(test_W_i, d0 = 1, d1 = shape_poly_2d[2])
    W_j = af.tile(test_W_j, d0 = 1, d1 = shape_poly_2d[2])

    P_xi_eta_quad_val = af.transpose(multivariable_poly_value(poly_xi_eta,
                                                              Xi, Eta))
    #P_xi_eta_quad_val = af.transpose(polyval_2d(poly_xi_eta, Xi, Eta))

    integral = af.sum(W_i * W_j * P_xi_eta_quad_val, dim = 0)

    return af.transpose(integral)


def lobatto_quad_multivar_poly(poly_xi_eta, N_quad, advec_var):
    '''
    '''
    shape_poly_2d = shape(poly_xi_eta)

    xi_LGL  = lagrange.LGL_points(N_quad)
    eta_LGL = lagrange.LGL_points(N_quad)

    Xi, Eta = af_meshgrid(xi_LGL, eta_LGL)

    Xi  = af.flat(Xi)
    Eta = af.flat(Eta)

    w_i = lagrange.lobatto_weights(N_quad)
    w_j = lagrange.lobatto_weights(N_quad)

    W_i, W_j = af_meshgrid(w_i, w_j)

    W_i = af.tile(af.flat(W_i), d0 = 1, d1 = shape_poly_2d[2])
    W_j = af.tile(af.flat(W_j), d0 = 1, d1 = shape_poly_2d[2])

    P_xi_eta_quad_val = af.transpose(polyval_2d(poly_xi_eta, Xi, Eta))

    integral = af.sum(W_i * W_j * P_xi_eta_quad_val, dim = 0)

    return af.transpose(integral)


def integrate_2d_multivar_poly(poly_xi_eta, N_quad, scheme, advec_var):
    '''
    Evaluates the integral

    .. math:: \\iint P(\\xi, \\eta) \\partial \\xi \\partial \\eta

    using 

    Parameters
    ----------
    poly_xi_eta : multivar_poly
                  The multivariable polynomial returned by the method
                  :py:meth:`dg_maxwell.utils.polynomial_product_coeffs`
                  Multiple polynomials can be returned, differnt polynomials
                  should be in the :math:`3^{rd}` dimension.

    N_quad  : int
              Order of the Gauss-Legendre or Gauss-Lobatto quadrature.

    scheme  : str
              Possible options are

              - ``gauss`` for using Gauss-Legendre quadrature
              - ``lobatto`` for using Gauss-Lobatto quadrature

    Returns
    -------
    integrate_poly_xi_eta : af.Array [number_of_polynomials 1 1 1]
                            Integral

                            .. math:: \\iint P(\\xi, \\eta) \\partial\\xi \\
                                      \\partial\\eta

    '''
    if scheme is 'gauss':
        return gauss_quad_multivar_poly(poly_xi_eta, N_quad, advec_var)

    elif scheme is 'lobatto':
        return lobatto_quad_multivar_poly(poly_xi_eta, N_quad, gv)

    else:
        return

    return


def centroid(x_nodes, y_nodes):
    '''
    Finds the centroid of a given :math:`2D` closed geometry.
    
    Parameters
    ----------
    x_nodes : np.array [N]
              :math:`x` nodes of the vertices of the geometry

    y_nodes : np.array [N]
              :math:`y` nodes of the vertices of the geometry

    Returns
    -------
        tuple(float, float)
        :math:`(x, y)` coordinates of the centroid
    '''
    return (np.mean(x_nodes), np.mean(y_nodes))
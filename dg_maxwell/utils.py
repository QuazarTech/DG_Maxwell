#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.lines as lines
import arrayfire as af
af.set_backend('opencl')
af.set_device(1)

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
          returns the sum of a and b. When used along with af.broadcast
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
        An array which contains 'number_of_points' evenly spaced points
        between 'start' and 'end'

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
                   :math:`(x, y)` coordinates of :math:`N` points. First and second
                   column stores :math:`x` and :math:`y` coordinates of an point.
             
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
        axes_handler.add_line(lines.Line2D(line1_xs, line1_ys, linewidth=grid_width, color=grid_color))
        
    return


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

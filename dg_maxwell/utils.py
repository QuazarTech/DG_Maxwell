#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import csv

import numpy as np
import matplotlib.lines as lines
import arrayfire as af

from dg_maxwell import params

af.set_backend(params.backend)

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
        axes_handler.add_line(lines.Line2D(line1_xs, line1_ys,
                                           linewidth=grid_width, color=grid_color))
        
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

    for n, line in enumerate(csv_handler):
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
    
    Arr_0 = af.data.tile(af.array.transpose(arr_0), d0 = arr_1.shape[0])
    Arr_1 = af.data.tile(arr_1, d0 = 1, d1 = arr_0.shape[0])
    
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
    a : af,Array [M N Q 1]
        First set of :math:`Q` :math:`2D` arrays.
        :math:`N \neq 1 & M \neq 1`

    b : af,Array [N P Q 1]
        Second set of :math:`Q` :math:`2D` arrays.
        :math:`P \neq 1`

    Returns
    -------
    matmul : af.Array [M P Q 1]
             Matrix multiplication of :math:`Q` sets of 2D arrays

    '''
    shape_a = shape(a)
    shape_b = shape(b)

    M = shape_a[0]
    N = shape_a[1]
    P = shape_b[1]
    Q = shape_a[2]
    
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


def polyval_2D(polynomials, x, y):
    '''
    Takes the 2D polynomials and finds it'their values at the :math:`(x, y)`
    coordinates.
    
    Parameters
    ----------
    polynomials: af.Array [number_of_polynomials N 1 1]
                 ``number_of_polynomials`` :math:`2D` polynomials of degree
                 :math:`N` of the form
                 
                 .. math:: P(x, y) = a_0y^Nx^0 + a_1y^{N - 1}x^1 + ... \\
                           a_2y^1x^{N - 1} + a_{N + 1}y^0x^N
                  
    x          : af.Array [Q P 1 1]
                 :math:`x` coordinates for which the polynomials are to be
                 evaluated
                 
    y          : af.Array [Q P 1 1]
                :math:`x` coordinates for which the polynomials are to be
                evaluated
    '''
    
    N = int(polynomials.shape[1])
    x = af.reorder(x, d0 = 2, d1 = 1, d2 = 0)
    y = af.reorder(y, d0 = 2, d1 = 1, d2 = 0)
    x = af.tile(x, d0 = N)
    y = af.tile(y, d0 = N)
    
    power_y = af.tile(af.range(N), d0 = 1, d1 = x.shape[1], d2 = y.shape[2])
    power_x = af.flip(power_y, dim = 0)
    
    x_power = x ** power_x
    y_power = y ** power_y
    
    xy = x_power * y_power
    
    return af.matmul(polynomials, xy)

def integrate_2d(polynomial, scheme = 'gauss'):
    '''
    Takes the coefficients of the polynomials as an argument and calculates
    the :math:`2D` integral using either Legendre-Gauss quadrature or
    Gauss-Lobatto quadrature.
    
    Parameters
    ----------
    polynomial : af.Array [number_of_polynomials N 1 1]
                 ``number_of_polynomials`` :math:`2D` polynomials of degree
                 :math:`N` of the form
                 
                 .. math:: P(x, y) = a_0y^Nx^0 + a_1y^{N - 1}x^1 + ... \\
                           a_2y^1x^{N - 1} + a_{N + 1}y^0x^N
                           
    scheme     : str
                 Quadrature scheme to be used for the numerical integration.
                 scheme could accept ``gauss`` and ``lobatto``
                 
    Returns
    -------
    af.Array[number_of_polynomials 1 1 1]
    Integral for each of the polynomials.
    '''
    
    
    
    return
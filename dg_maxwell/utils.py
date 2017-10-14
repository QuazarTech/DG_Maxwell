#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import csv

import numpy as np
import matplotlib.lines as lines
import arrayfire as af
af.set_backend('cpu')

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
    arrayfire.Array
          returns the sum of a and b. When used along with af.broadcast
          can be used to sum different size arrays.
    '''
    return a + b


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
    quotient : arrayfire.Array
               returns the quotient a / b. When used along with af.broadcast
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
    product : arrayfire.Array
              returns the quotient a / b. When used along with af.broadcast
              can be used to give quotient of two different size arrays
              by multiplying elements of the broadcasted array.
    '''
    product = a* b
    
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
    arrayfire.Array
              returns the quotient a / b. When used along with af.broadcast
              can be used to give quotient of two different size arrays
              by multiplying elements of the broadcasted array.
    '''
    return a ** b



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
    a_n1 = a.shape[0]
    a_n2 = a.shape[1]

    b_n1 = b.shape[0]
    b_n2 = b.shape[1]
    
    a_reorder = af.reorder(a, d0 = 0, d1 = 2, d2 = 1)
    b_reorder = af.reorder(b, d0 = 0, d1 = 2, d2 = 1)
    b_reorder = af.transpose(b_reorder)
    
    a_tile = af.tile(a_reorder, d0 = 1, d1 = b.shape[0])
    b_tile = af.tile(b_reorder, d0 = a.shape[0])
    
    return a_tile * b_tile

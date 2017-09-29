#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.lines as lines

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


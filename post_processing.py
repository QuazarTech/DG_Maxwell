#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from matplotlib import pyplot as pl
from tqdm import trange
import h5py
import numpy as np
import arrayfire as af

from dg_maxwell import params
from dg_maxwell import global_variables

pl.rcParams['figure.figsize'  ] = 9.6, 6.
pl.rcParams['figure.dpi'      ] = 300
pl.rcParams['image.cmap'      ] = 'jet'
pl.rcParams['lines.linewidth' ] = 1.5
pl.rcParams['font.family'     ] = 'serif'
pl.rcParams['font.weight'     ] = 'bold'
pl.rcParams['font.size'       ] = 20
pl.rcParams['font.sans-serif' ] = 'serif'
pl.rcParams['text.usetex'     ] = True
pl.rcParams['axes.linewidth'  ] = 1.5
pl.rcParams['axes.titlesize'  ] = 'medium'
pl.rcParams['axes.labelsize'  ] = 'medium'
pl.rcParams['xtick.major.size'] = 8
pl.rcParams['xtick.minor.size'] = 4
pl.rcParams['xtick.major.pad' ] = 8
pl.rcParams['xtick.minor.pad' ] = 8
pl.rcParams['xtick.color'     ] = 'k'
pl.rcParams['xtick.labelsize' ] = 'medium'
pl.rcParams['xtick.direction' ] = 'in'
pl.rcParams['ytick.major.size'] = 8
pl.rcParams['ytick.minor.size'] = 4
pl.rcParams['ytick.major.pad' ] = 8
pl.rcParams['ytick.minor.pad' ] = 8
pl.rcParams['ytick.color'     ] = 'k'
pl.rcParams['ytick.labelsize' ] = 'medium'
pl.rcParams['ytick.direction' ] = 'in'

gv = global_variables.advection_variables(params.N_LGL, params.N_quad,\
                                          params.x_nodes, params.N_Elements,\
                                          params.c, params.total_time, params.wave,\
                                          params.c_x, params.c_y, params.courant,\
                                          params.mesh_file, params.total_time_2d)

#print(advection_2d.time_evolution(gv))
gauss_points    = gv.gauss_points
gauss_weights   = gv.gauss_weights
xi_LGL          = gv.xi_LGL
lagrange_coeffs = gv.lagrange_coeffs
Li_Lj_coeffs    = gv.Li_Lj_coeffs
u               = gv.u_e_ij
lobatto_weights = gv.lobatto_weights_quadrature
x_e_ij          = gv.x_e_ij
y_e_ij          = gv.y_e_ij
delta_t_2d      = gv.delta_t_2d


def contour_2d(u, index):
    '''
    '''
    color_levels = np.linspace(-1.1, 1.1, 100)
    u_plot = af.flip(af.moddims(u, params.N_LGL, params.N_LGL, 10, 10), 0)
    x_plot = af.flip(af.moddims(x_e_ij, params.N_LGL, params.N_LGL, 10, 10), 0)
    y_plot = af.flip(af.moddims(y_e_ij, params.N_LGL, params.N_LGL, 10, 10), 0)


    x_contour = af.np_to_af_array(np.zeros([params.N_LGL * 10, params.N_LGL * 10]))
    y_contour = af.np_to_af_array(np.zeros([params.N_LGL * 10, params.N_LGL * 10]))
    u_contour = af.np_to_af_array(np.zeros([params.N_LGL * 10, params.N_LGL * 10]))
    fig = pl.figure()
    #
    for i in range(100):
        p = int(i / 10)
        q = i - p * 10
        x_contour[p * params.N_LGL:params.N_LGL * (p + 1),\
                  q * params.N_LGL:params.N_LGL * (q + 1)] = x_plot[:, :, q, p]

        y_contour[p * params.N_LGL:params.N_LGL * (p + 1),\
                  q * params.N_LGL:params.N_LGL * (q + 1)] = y_plot[:, :, q, p]

        u_contour[p * params.N_LGL:params.N_LGL * (p + 1),\
                  q * params.N_LGL:params.N_LGL * (q + 1)] = u_plot[:, :, q, p]

    x_contour = np.array(x_contour)
    y_contour = np.array(y_contour)
    u_contour = np.array(u_contour)
    pl.contourf(x_contour, y_contour, u_contour, 200, levels = color_levels, cmap = 'jet')
    pl.gca().set_aspect('equal')
    pl.colorbar()
    pl.title('Time = %.2f' %(index * 10 * delta_t_2d))
    fig.savefig('results/2D_Wave_images/%04d' %(index) + '.png')
    pl.close('all')
    return
           

for i in trange(201):
    h5py_data = h5py.File('results/xi_eta_2d_hdf5_%02d/dump_timestep_%06d' %(int(params.N_LGL), int(10 * i)) + '.hdf5', 'r')
    u_LGL     = af.np_to_af_array(h5py_data['u_i'][:])
    contour_2d(u_LGL, i)
    if i > 199 :
        print(af.mean(af.abs(u_LGL - params.u_e_ij)))

# Creating a folder to store hdf5 files. If it doesn't exist.
results_directory = 'results/1D_Wave_images'

if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# The directory where h5py files are stored.
#h5py_directory = 'results/hdf5_%02d' %int(params.N_LGL)
#
#path, dirs, files = os.walk(h5py_directory).__next__()
#file_count = len(files)
#print(file_count)
#
#
#for i in range(0, file_count):
#    fig = pl.figure()
#    h5py_data = h5py.File('results/hdf5_%02d/dump_timestep_%06d' %(int(params.N_LGL), int(20 * i)) + '.hdf5', 'r')
#    u_LGL     = (h5py_data['u_i'][:])
#    pl.plot(params.element_LGL, u_LGL)
#    pl.xlabel('x')
#    pl.ylabel('u')
#    pl.xlim(-1, 1)
#    pl.ylim(-2, 2)
#    pl.title('Time = %f' % (i * 20 * params.delta_t))
#    fig.savefig('results/1D_Wave_images/%04d' %(i) + '.png')
#    pl.close('all')
#
## Creating a movie with the images created.
#os.system("cd results/1D_Wave_images && ffmpeg -f image2 -i %04d.png -vcodec mpeg4\
#	  -mbd rd -trellis 2 -cmp 2 -g 300 -pass 1 -r 25 -b 18000000 movie.mp4\
#	  && rm -f ffmpeg2pass-0.log\
#          && mv movie.mp4 1D_wave_advection.mp4\
#          && mv 1D_wave_advection.mp4 ~/git/DG_Maxwell/results && rm *")

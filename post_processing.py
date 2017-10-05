#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from matplotlib import pyplot as pl
import arrayfire as af
import numpy as np
import h5py

from dg_maxwell import params

pl.rcParams['figure.figsize'  ] = 9.6, 6.
pl.rcParams['figure.dpi'      ] = 100
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






# Creating a folder to store hdf5 files. If it doesn't exist.
results_directory = 'results/1D_Wave_images'

if not os.path.exists(results_directory):
    os.makedirs(results_directory)


N = os.system('cd results/hdf5 && find -maxdepth 1 -type f | wc -l')



for i in range(0,int( N)):
    print('?')
    fig = pl.figure()
    h5py_data = h5py.File('results/hdf5/dump_timestep_%06d'%(20 * i) + '.hdf5', 'r')
    u_LGL     = (h5py_data['u_i'][:])
    pl.plot(params.element_LGL, u_LGL)
    pl.xlabel('x')
    pl.ylabel('u')
    pl.title('Time = %f' % (i * 20))
    fig.savefig('results/1D_Wave_images/%04d' %(i) + '.png')
    pl.close('all')


fig = pl.figure()
h5py_data = h5py.File('results/hdf5/dump_timestep_%06d'%(20 * 2) + '.hdf5', 'r')
u_LGL     = (h5py_data['u_i'][:])
pl.plot(params.element_LGL, u_LGL)
pl.show()
fig.savefig('results/1D_Wave_images/%04d' %2 + '.png')
pl.close()

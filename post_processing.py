#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from matplotlib import pyplot as pl
import arrayfire as af
import numpy as np
import h5py

from dg_maxwell import params

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




# Creating a folder to store hdf5 files. If it doesn't exist.
results_directory = 'results/1D_Wave_images'

if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# The directory where h5py files are stored.
h5py_directory = 'results/hdf5_%02d' %int(params.N_LGL)

path, dirs, files = os.walk(h5py_directory).__next__()
file_count = len(files)
print(file_count)


for i in range(0, file_count):
    fig = pl.figure()
    h5py_data = h5py.File('results/hdf5_%02d/dump_timestep_%06d' %(int(params.N_LGL), int(20 * i)) + '.hdf5', 'r')
    u_LGL     = (h5py_data['u_i'][:])
    pl.plot(params.element_LGL, u_LGL)
    pl.xlabel('x')
    pl.ylabel('u')
    pl.xlim(-1, 1)
    pl.ylim(-2, 2)
    pl.title('Time = %f' % (i * 20 * params.delta_t))
    fig.savefig('results/1D_Wave_images/%04d' %(i) + '.png')
    pl.close('all')

# Creating a movie with the images created.
os.system("cd results/1D_Wave_images && ffmpeg -f image2 -i %04d.png -vcodec mpeg4\
	  -mbd rd -trellis 2 -cmp 2 -g 300 -pass 1 -r 25 -b 18000000 movie.mp4\
	  && rm -f ffmpeg2pass-0.log\
          && mv movie.mp4 1D_wave_advection.mp4\
          && mv 1D_wave_advection.mp4 ~/git/DG_Maxwell/results && rm *")

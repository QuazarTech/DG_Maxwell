import arrayfire as af
import numpy as np
from matplotlib import pyplot as pl

from dg_maxwell import params
from dg_maxwell import lagrange
from dg_maxwell import wave_equation_2d
from dg_maxwell import wave_equation
from dg_maxwell import isoparam
from dg_maxwell import msh_parser


#u_pq = isoparam.u_pq_isoparam()
#print(u_pq[:, :, 5])
element_LGL = wave_equation_2d.u_init()
print(af.reorder(element_LGL[0, 0, :, :], 2, 3, 1, 0))
print(element_LGL.shape)

def g_dd(x_nodes, y_nodes, xi, eta):
   """
   """
   
   ans00  =   (wave_equation_2d.dx_xi(x_nodes, xi, eta))**2 \
            + (wave_equation_2d.dy_xi(y_nodes, xi, eta))**2
   ans11  =   (wave_equation_2d.dx_eta(x_nodes, xi, eta))**2 \
            + (wave_equation_2d.dy_eta(y_nodes, xi, eta))**2
   
   ans01  =  (wave_equation_2d.dx_xi(x_nodes, xi, eta))  \
           * (wave_equation_2d.dx_eta(x_nodes, xi, eta)) \
           + (wave_equation_2d.dy_xi(y_nodes, xi, eta))  \
           * (wave_equation_2d.dy_eta(y_nodes, xi, eta))
   
   ans =  [[ans00, ans01],
           [ans01, ans11]
          ]
   
   return np.array(ans)

def g_uu(x_nodes, y_nodes, xi, eta):
   gCov = g_dd(x_nodes, y_nodes, xi, eta)
   
   
   a = gCov[0][0]
   b = gCov[0][1]
   c = gCov[1][0]
   d = gCov[1][1]
   
   det = (a*d - b*c)
   
   ans = [[d, -b],
           [-c, a]
          ]
   return np.array(ans)/det


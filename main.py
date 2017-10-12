from dg_maxwell import wave_equation
from dg_maxwell import params
from dg_maxwell import lagrange

u_diff = wave_equation.time_evolution()
print(wave_equation.L1_norm(u_diff))

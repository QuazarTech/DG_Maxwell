from dg_maxwell import params
from dg_maxwell import wave_equation

if __name__ == '__main__':
    wave_equation.time_evolution()
    print(params.volume_integrand_N_LGL)

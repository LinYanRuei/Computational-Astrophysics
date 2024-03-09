import numpy as np
import matplotlib.pyplot as plt


result_array = []
n_time = 100


for n in range(n_time):
    L   = 1.0   # 1-D computational domain size
    N   = n+50   # number of computing cells
    v   = 1.0   # advection velocity
    u0  = 1.0   # background density
    amp = 0.5   # sinusoidal amplitude
    cfl = 0.9   # Courant condition factor

    # derived constants
    dx     = L/N               # spatial resolution
    dt     = cfl*dx/np.abs(v)  # time interval for data update
    period = L/np.abs(v)       # time period
    def ref_func( x, t ):
        k = 2.0*np.pi/L   # wavenumber
        return u0 + amp*np.sin( k*(x-v*t) )

    t = 0.0
    x = np.arange( 0.0, L, dx ) + 0.5*dx   # cell-centered coordinates
    u = ref_func( x, t )                   # initial density distribution

    t = 0.0
    x = np.arange( 0.0, L, dx ) + 0.5*dx   # cell-centered coordinates
    u = ref_func( x, t )                   # initial density distribution
    end_time        = 2.0*period  # simulation time
    nstep_per_image = 1           # plotting frequency

    def evolution(time):
        for step in range( nstep_per_image ):
            global u,t

            u_in = u.copy()

        #     calculate the half-timestep solution
            u_half = np.empty( N )
            for i in range( N ):
        #       u_half[i] is defined at the left face of cell i
                im = (i-1+N) % N  # assuming periodic boundary condition
                ip = (i+1) % N
                u[i] = u_in[i] - dt*v*( u_in[i] - u_in[im] )/dx

        #     update time
            t = t + dt

            if ( t >= end_time ):   break

    #  calculate the reference analytical solution and estimate errors
        u_ref = ref_func( x, t )
        err   = np.abs( u_ref - u ).sum()/N
        return err
    for j in range(int( np.ceil( end_time/(nstep_per_image*dt) ) )):
        result=(evolution(j))
    result_array.append(result)


N_array = np.linspace(1,n_time,n_time)
N_array = np.log(N_array+50)
result_array = np.log(result_array)
coeffs = np.polyfit(N_array, result_array, 1)
slope = coeffs[0]
intercept = coeffs[1]
y_fit = slope * N_array + intercept
plt.plot(N_array, result_array, color = 'blue', label = 'Numerical result')
plt.plot(N_array, y_fit, color='red', label = 'fitting line',linestyle='dashed')
plt.title(f"Slope:{slope:.1f}")
plt.xlabel('log(number of computing cells)')
plt.ylabel('log(error)')
plt.legend(loc = 'best')
plt.show()
import numpy as np
import matplotlib.pyplot as plt


result_array = []
n_time = 10
err_array = []

for n in range(n_time):
    L   = 1.0   # 1-D computational domain size
    N   = 10*n+40   # number of computing cells
    D   = 1.0   # diffusion coefficient
    v   = 1.0   # advection velocity
    u0  = 1.0   # background density
    amp = 0.5   # sinusoidal amplitude
    cfl = 0.9   # Courant condition factor

    # derived constants
    dx      = L/(N-1)                # spatial resolution
    dt      = cfl*0.5*dx**2.0/D      # time interval for data update
    t_scale = (0.5*L/np.pi)**2.0/D   # diffusion time scale across L
    def ref_func( x, t ):
        k = 2.0*np.pi/L   # wavenumber
        return u0 + amp*np.sin( k*(x-v*t) )

    t = 0.0
    x = np.linspace( 0.0, L, N )   # cell-centered coordinates
    u = ref_func( x, t )                   # initial density distribution
    # set the coefficient matrices A with A*u(t+dt)=u(t)
    r = D*dt/dx**2/2   #musr=t change r /2
    A = np.diagflat( np.ones(N-3)*(-r),       -1 ) + \
        np.diagflat( np.ones(N-2)*(1.0+2.0*r), 0 ) + \
        np.diagflat( np.ones(N-3)*(-r),       +1 );


    B = np.diagflat( np.ones(N-3)*(r),       -1 ) + \
        np.diagflat( np.ones(N-2)*(1.0-2.0*r), 0 ) + \
        np.diagflat( np.ones(N-3)*(r),       +1 );
    # plotting parameters
    
    end_time        = 2.0*t_scale  # simulation time
    nstep_per_image = 1           # plotting frequency

    def evolution(time):
        global t, u
        for step in range( nstep_per_image ):
        #     update all **interior** cells with the BTCS scheme
        #     by solving A*u(t+dt) = u(t)p

        #     (1) copy u(t) for adding boundary conditions
            u_bk = np.copy( u[1:-1] )
            u_bk = np.dot(B, u_bk)
        #     (2) apply the Dirichlet boundary condition: u[0]=u[N-1]=u0
            u_bk[ 0] += 2*r*u0    #constant of 2 because of BC
            u_bk[-1] += 2*r*u0

        #     (3) compute u(t+dt)

            u[1:-1] = np.linalg.solve( A, u_bk )

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
# N_array = np.log(N_array+50)
# result_array = np.log(result_array)
coeffs = np.polyfit(N_array, result_array, 1)
slope = coeffs[0]
intercept = coeffs[1]
y_fit = slope * N_array + intercept
plt.scatter(10*N_array+40, result_array, color = 'blue', label = 'Numerical result')
plt.plot(10*N_array+40, y_fit, color='red', label = 'fitting line',linestyle='dashed')
plt.title(f"Slope:{slope:.5f}")
plt.xlabel('number of computing cells')
plt.ylabel('error')
plt.legend(loc = 'best')
plt.show()
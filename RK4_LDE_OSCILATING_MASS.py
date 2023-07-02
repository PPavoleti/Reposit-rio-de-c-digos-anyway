import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def oscilator(y, t):
    """    
    Defines the ODE and it's parameters.
    
    Parameters:
    - y: Array containing xi and vi values
    - t: float with the interaction instant
    
    Returns:
    - A array with evaluated dxidt dvidt

    NOTE:
    - TODO: rewrite the function so all coeficients
    could be given by a input array and define dxidt
    and dvidt for any number of particles.
    """
    
    x1, x2, v1, v2 = y
    
    m1 = 1.0                # 1st particle's mass
    m2 = 1.0                # 2nd particle's mass
    k1 = 2.5                # elasticity coefficient 1
    k2 = 5.0                # elasticity coefficient 2
    k3 = 4.0                # elasticity coefficient 3
    b1 = 0.1                # damping coefficient 1
    b2 = 1.0                # damping coefficient 2
    A = 5.0                 # force amplitude
    w = 2.0                 # oscilating constant omega
    F = -A * np.cos(w*t)    # oscilating force
    
    dx1dt = v1
    dx2dt = v2
    dv1dt = -(k1 * x1 - k2 * (x2 - x1)) / m1 - F / m1 - b1 * v1 
    dv2dt = -(k2 * (x2 - x1) + k3 * x2) / m2 - b2 * v2

    return np.array([dx1dt, dx2dt, dv1dt, dv2dt]) 


def rk4(A, y0, t0, t1, h):
    """
    Solve a system of linear differential equations using the RK4 method.

    Parameters:
    - dydt: A function that calculates the derivative of the system.
            It takes the current state vector y and time t as inputs and returns the derivative dy/dt.
            The derivative should be returned as a 1D numpy array.
    - y0: The initial state vector at t0.
          It should be given as a 1D numpy array.
    - t0: The initial time.
    - t1: The final time.
    - h: The step size.

    Returns:
    - t: A 1D numpy array containing the time values from t0 to t1 with step h.
    - y: A 2D numpy array containing the state vectors corresponding to each time point.
         Each row represents a state vector at a specific time.
         
    Complexity:
        - Considering N = num_steps = (t1 - t0) / h + 1
        we can say that the complexity of this algorithm
        is of the order O(N) since the RK4 method mainly 
        depends on the number of steps performed in each 
        iteration, calculating k1, k2, k3, and k4.
    
    """
    
    num_step = int((t1+t0)/h)+1
    t = np.linspace(t0, t1, num_step)
    y = np.zeros((num_step, len(y0)))
    y[0] = y0
    
    
    for i in range(num_step - 1):
        k1 = h * oscilator(y[i] , t[i])
        k2 = h * oscilator(y[i] + 0.5 * k1 , t[i] + 0.5 * h)
        k3 = h * oscilator(y[i] + 0.5 * k2 , t[i] + 0.5 * h)
        k4 = h * oscilator(y[i] + k3 , t[i] + h)
        
        y[i+1] = y[i] + (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)
        
    return t, y


# initial conditions
y0 = np.array([10.0, 0.0, 0.0, 0.0])  # [x1, x2, v1, v2]
t0 = 0.0
t1 = 100.0
h = 0.1
# numerical solution by using the RK4 method.
t, y = rk4(oscilator, y0, t0, t1, h)
for ti, xi in zip(t, y):
    print(f"t = {ti:.2f}, y = {xi}")
# plot section
sns.set_theme()
plt.plot(t, y[:, 0], label='X1')
plt.plot(t, y[:, 1], label='X2')
plt.plot(t, y[:, 2], label='V1')
plt.plot(t, y[:, 3], label='V2')
plt.xlabel('Tempo - (s)')
plt.ylabel('Posição - (m)')
plt.title('Sistema de Osciladores Acoplados (RK4)')
plt.legend()
plt.grid(True)
plt.show()

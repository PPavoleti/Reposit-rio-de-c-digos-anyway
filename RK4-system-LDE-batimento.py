# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 00:35:15 2023

@author: paulo
"""

import numpy as np
import matplotlib.pyplot as plt


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
    
    num_step=int((t1+t0)/h)+1
    t=np.linspace(t0, t1, num_step)
    y=np.zeros((num_step, len(y0)))
    y[0]=y0
    
    
    for i in range(num_step - 1):
        k1 = h * sis_osc_for(y[i] , t[i])
        k2 = h * sis_osc_for(y[i] + 0.5 * k1 , t[i] + 0.5 * h)
        k3 = h * sis_osc_for(y[i] + 0.5 * k2 , t[i] + 0.5 * h)
        k4 = h * sis_osc_for(y[i] + k3 , t[i] + h)
        
        y[i+1] = y[i] + (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)
        
    return t, y

def sis_osc_for ( y, t):
    x1, x2, v1, v2 = y
    m1 = 1.0  # massa do oscilador 1
    m2 = 1.0  # massa do oscilador 2
    k1 = 2.5  # constante de elasticidade do oscilador 1
    k2 = 5.0  # constante de elasticidade do oscilador 2
    k3 = 4.0
    b1 = 0.1   # coeficiente de amortecimento para o corpo 1
    b2 = 1.0    # coeficiente de amortecimento para o corpo 2
    p = 5.0   #máximo da força oscilante
    w = 2.0   #constante de oscilação da força oscilante
    F = -p*np.sin(w*t) #força oscilante
    dx1dt = v1
    dx2dt = v2
    dv1dt = -(k1 * x1 - k2 * (x2 - x1)) / m1 - F/m1- b1 * v1 
    dv2dt = -(k2 * (x2 - x1) + k3 * x2) / m2 - b2 * v2

    return np.array([dx1dt, dx2dt, dv1dt, dv2dt]) 

# Condições iniciais
y0 = np.array([10.0, 0.0, 0.0, 0.0])  # [x1, x2, v1, v2]
t0 = 0.0
t1 = 100.0
h = 0.1

# Solução numérica utilizando o método RK4
t, y = rk4(sis_osc_for, y0, t0, t1, h)

for ti, xi in zip(t, y):
    print(f"t = {ti:.2f}, y = {xi}")

# Plot
plt.plot(t, y[:, 0], label='x1')
plt.plot(t, y[:, 1], label='x2')
#plt.plot(t, y[:, 2], label='v1')
#plt.plot(t, y[:, 3], label='v2')
plt.xlabel('Tempo')
plt.ylabel('Posição')
plt.title('Sistema de Osciladores Acoplados (RK4)')
plt.legend()
plt.grid(True)
plt.show()
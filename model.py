import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def model(t, y, V, k, F):
    C_A, C_B = y
    r = k * C_A
    dC_A_dt = F/V * (1.1 - C_A) - r
    dC_B_dt = r - F/V * C_B  # Assuming B is also carried out by the flow at the same rate
    return [dC_A_dt, dC_B_dt]

# Parameters
V = 10      # Volume in m^3
k = 0.2     # Reaction rate constant in 1/s
F = 0.5     # Volumetric flow rate in m^3/s

# Initial Conditions
C_A0 = 1.1  # Initial concentration of A in mol/m^3
C_B0 = 0.0  # Initial concentration of B in mol/m^3

# Time span for the simulation
t_span = (0, 30)  # Time from 0 to 30 seconds
t_eval = np.linspace(t_span[0], t_span[1], 300)

# Solve the system of ODEs
result = solve_ivp(model, t_span, [C_A0, C_B0], args=(V, k, F), t_eval=t_eval, method='RK45')

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(result.t, result.y[0], label='Concentration of A (mol/m³)')
plt.plot(result.t, result.y[1], label='Concentration of B (mol/m³)')
plt.title('Concentration of A and B in a CSTR')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mol/m³)')
plt.legend()
plt.grid(True)
plt.show()

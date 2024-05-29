import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Constants
k = 0.2  # reaction rate constant, 1/s
V = 10.0  # volume of reactor, m^3
F = 0.5  # volumetric flow rate, m^3/s
CA0 = 1.1  # initial concentration of A, mol/m^3
CB0 = 0.0  # initial concentration of B, mol/m^3
T0 = 298.0  # initial temperature, K
T_in = 298.0  # inlet temperature, K
UA = 200.0  # overall heat transfer coefficient, J/s-K
Cp = 4.0  # heat capacity, J/mol-K
deltaH = -50000.0  # heat of reaction, J/mol
rho = 1000.0  # density, kg/m^3
Cp_rho = Cp * rho

# Define the system of ODEs
def odes(y, t):
    CA, CB, T = y
    r = k * CA
    dCAdt = (CA0 - CA) * F / V - r
    dCBdt = r - CB * F / V
    dTdt = ((T_in - T) * F * Cp_rho / V - UA * (T - T_in) / V + (-deltaH * r)) / (Cp_rho / V)
    return [dCAdt, dCBdt, dTdt]

# Initial conditions
y0 = [CA0, CB0, T0]

# Time points
t = np.linspace(0, 30, 500)

# Solve ODEs
solution = odeint(odes, y0, t)
CA = solution[:, 0]
CB = solution[:, 1]
T = solution[:, 2]

# Plot concentration of A
plt.figure()
plt.plot(t, CA, label='CA')
plt.xlabel('Time, s')
plt.ylabel('Concentration of A, mol/m^3')
plt.title('Concentration of A over Time')
plt.legend()
plt.grid()
plt.show()

# Plot concentration of B
plt.figure()
plt.plot(t, CB, label='CB')
plt.xlabel('Time, s')
plt.ylabel('Concentration of B, mol/m^3')
plt.title('Concentration of B over Time')
plt.legend()
plt.grid()
plt.show()

# Plot temperature
plt.figure()
plt.plot(t, T, label='Temperature')
plt.xlabel('Time, s')
plt.ylabel('Temperature, K')
plt.title('Temperature over Time')
plt.legend()
plt.grid()
plt.show()

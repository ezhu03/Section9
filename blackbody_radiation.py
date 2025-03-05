import numpy as np
from scipy.integrate import fixed_quad, quad

# Define the integrand after variable substitution:
def integrand(z):
    # z in [0,1). x = z/(1-z) and dx = dz/(1-z)^2.
    return (z**3) / ((1 - z)**5 * (np.exp(z/(1-z)) - 1))

# Numerically evaluate the integral over z from 0 to 1:
I, error = fixed_quad(integrand, 0, 1)

# Given constants:
k_B   = 1.38064852e-23  # Boltzmann constant in J/K
h     = 6.626e-34       # Planck's constant in J·s
c     = 3e8             # Speed of light in m/s
hbar  = h / (2 * np.pi) # Reduced Planck's constant

# Using the alternative prefactor form:
prefactor = k_B**4 / (c**2 * hbar**3 * 4*np.pi**2)

# Compute the Stefan-Boltzmann constant:
sigma = prefactor * I

print("Calculated Stefan-Boltzmann constant, σ =", sigma, "W/m²K⁴")
print("Expected result: σ ≈ 5.6704e-08 W/m²K⁴")

def integrandx(x):
    # z in [0,1). x = z/(1-z) and dx = dz/(1-z)^2.
    return (x**3) / (np.exp(x)-1)

I, error = quad(integrandx, 0, np.inf)
sigma = prefactor * I
print("Calculated Stefan-Boltzmann constant, σ =", sigma, "W/m²K⁴")
print("Expected result: σ ≈ 5.6704e-08 W/m²K⁴")
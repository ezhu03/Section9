import numpy as np
import matplotlib.pyplot as plt

# --- Parameters and Initial Conditions ---
e = 0.6
# Initial position and velocity (p = dq/dt)
q1_0 = 1 - e
q2_0 = 0.0
p1_0 = 0.0
p2_0 = np.sqrt((1 + e) / (1 - e))  # equals 2 for e = 0.6

# --- Explicit Euler Method (Task 2A) ---
Tf_A = 200.0          # final time
N_A = 100000          # number of steps
dt_A = Tf_A / N_A     # time step

# Initialize arrays for positions and momenta
q1_exp = np.zeros(N_A + 1)
q2_exp = np.zeros(N_A + 1)
p1_exp = np.zeros(N_A + 1)
p2_exp = np.zeros(N_A + 1)

# Set initial conditions
q1_exp[0] = q1_0
q2_exp[0] = q2_0
p1_exp[0] = p1_0
p2_exp[0] = p2_0

# Time-stepping loop for explicit Euler method
for i in range(N_A):
    # Compute the distance r and acceleration components
    r = np.sqrt(q1_exp[i]**2 + q2_exp[i]**2)
    a1 = -q1_exp[i] / r**3
    a2 = -q2_exp[i] / r**3

    # Update positions using current velocity
    q1_exp[i + 1] = q1_exp[i] + dt_A * p1_exp[i]
    q2_exp[i + 1] = q2_exp[i] + dt_A * p2_exp[i]
    
    # Update momenta using current acceleration
    p1_exp[i + 1] = p1_exp[i] + dt_A * a1
    p2_exp[i + 1] = p2_exp[i] + dt_A * a2

# --- Symplectic Euler Method (Task 2B) ---
Tf_B = 200.0          # final time
N_B = 400000          # number of steps
dt_B = Tf_B / N_B     # time step

# Initialize arrays for positions and momenta
q1_sym = np.zeros(N_B + 1)
q2_sym = np.zeros(N_B + 1)
p1_sym = np.zeros(N_B + 1)
p2_sym = np.zeros(N_B + 1)

# Set initial conditions
q1_sym[0] = q1_0
q2_sym[0] = q2_0
p1_sym[0] = p1_0
p2_sym[0] = p2_0

# Time-stepping loop for symplectic Euler method
for i in range(N_B):
    # Compute the distance r using the current position
    r = np.sqrt(q1_sym[i]**2 + q2_sym[i]**2)
    a1 = -q1_sym[i] / r**3
    a2 = -q2_sym[i] / r**3

    # Update momentum first ("kick")
    p1_sym[i + 1] = p1_sym[i] + dt_B * a1
    p2_sym[i + 1] = p2_sym[i] + dt_B * a2

    # Then update position using the updated momentum ("drift")
    q1_sym[i + 1] = q1_sym[i] + dt_B * p1_sym[i + 1]
    q2_sym[i + 1] = q2_sym[i] + dt_B * p2_sym[i + 1]

# --- Plotting the Orbits ---
plt.figure(figsize=(8, 8))
plt.plot(q1_exp, q2_exp, label='Explicit Euler (100,000 steps)')
plt.plot(q1_sym, q2_sym, label='Symplectic Euler (400,000 steps)', linestyle='--')
plt.xlabel('$q_1$')
plt.ylabel('$q_2$')
plt.title('Planet Orbit using Explicit and Symplectic Euler Methods')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
plt.savefig('orbits.png')
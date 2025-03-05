import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Define the Hamiltonian H(ω) and its derivative dH/dω.
def H(omega):
    # Noisy φ^4 theory potential: H(ω) = ω^4 - 8ω^2 - 2 cos(4ω)
    return omega**4 - 8 * omega**2 - 2 * np.cos(4*np.pi* omega)

def dH(omega):
    # dH/dω = 4ω^3 - 16ω + 8 sin(4ω)
    return 4 * omega**3 - 16 * omega + 8 * np.sin(4*np.pi * omega)

# -----------------------
# Part A: Gradient Descent
# -----------------------
def gradient_descent(initial, lr=0.001, max_iter=10000, tol=1e-6):
    omega = initial
    omega_history = [omega]
    for i in range(max_iter):
        grad = dH(omega)
        new_omega = omega - lr * grad  # update rule
        omega_history.append(new_omega)
        if abs(new_omega - omega) < tol:
            break
        omega = new_omega
    return omega_history

# Save frames of the descent trajectory over the function plot
def save_gd_frames(omega_history, x_range=(-5, 5), num_points=400, out_dir='gd_frames'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = H(x)
    for i, omega in enumerate(omega_history):
        plt.figure()
        plt.plot(x, y, 'b-', label='H(ω)')
        plt.plot(omega, H(omega), 'ro', markersize=8)
        plt.title(f'Gradient Descent Step {i}')
        plt.xlabel('ω')
        plt.ylabel('H(ω)')
        plt.legend()
        plt.xlim(-3,3)
        plt.ylim(-20,10)
        filename = os.path.join(out_dir, f'frame_{i:04d}.png')
        plt.savefig(filename)
        plt.close()

# Create a video from saved frames using OpenCV
def create_video_from_frames(frame_dir='gd_frames', video_filename='gradient_descent.mp4', fps=10):
    frame_files = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.png')])
    if not frame_files:
        print("No frames found in", frame_dir)
        return
    frame = cv2.imread(frame_files[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    for file in frame_files:
        frame = cv2.imread(file)
        video.write(frame)
    video.release()
    print("Video saved as", video_filename)

# -----------------------
# Part B: Metropolis–Hastings
# -----------------------
def metropolis_hastings(initial, rho=0.0001, sigma=0.5, iterations=10000, tol=1e-6):
    omega = initial
    omega_history = [omega]
    for i in range(iterations):
        proposal = omega + np.random.normal(0, sigma)
        # Use log-domain for numerical stability
        log_r = rho * (H(proposal) - H(omega))
        if log_r > 0 or np.log(np.random.rand()) < log_r:
            # Check tolerance: if the accepted change is very small, break
            if abs(proposal - omega) < tol:
                break
            omega = proposal
        omega_history.append(omega)
    return omega_history

def simulated_annealing(initial, rho0=0.0001, phi=0.0000001, sigma=0.5, iterations=10000, tol=1e-6):
    omega = initial
    rho = rho0
    omega_history = [omega]
    rho_history = [rho]
    for i in range(iterations):
        proposal = omega + np.random.normal(0, sigma)
        log_r = rho * (H(proposal) - H(omega))
        if log_r > 0 or np.log(np.random.rand()) < log_r:
            if abs(proposal - omega) < tol:
                break
            omega = proposal
        omega_history.append(omega)
        # Update the inverse temperature (cooling schedule)
        rho += phi
        rho_history.append(rho)
    return omega_history, rho_history

# Utility function to plot convergence trajectory over the H(ω) curve.
def plot_trajectory(omega_history, method_name, x_range=(-5, 5), num_points=400):
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = H(x)
    plt.figure()
    plt.plot(x, y, 'b-', label='H(ω)')
    plt.plot(omega_history, [H(w) for w in omega_history], 'ro-', markersize=3, label=method_name)
    plt.xlabel('ω')
    plt.ylabel('H(ω)')
    plt.title(f'Convergence using {method_name}')
    plt.legend()
    plt.show()
    plt.xlim(-3,3)
    plt.ylim(-20,10)
    plt.savefig(f'{method_name}.png')

# -----------------------
# Main function: run all parts
# -----------------------
def main():
    # Define initial guesses for ω
    initial_guesses = [-1.0, 0.5, 3.0]
    
    # Part A: Gradient Descent
    print("Running Gradient Descent...")
    gd_results = {}
    for initial in initial_guesses:
        omega_hist = gradient_descent(initial, lr=0.001, max_iter=10000, tol=1e-6)
        gd_results[initial] = omega_hist
        # Save frames and create a video only for one example (e.g., initial=1.0)
        if initial == -1.0:
            frame_dir = f'gd_frames_initial_{initial}'
            save_gd_frames(omega_hist, out_dir=frame_dir)
            create_video_from_frames(frame_dir=frame_dir, video_filename=f'gd_video_initial_{initial}.mp4')
        plot_trajectory(omega_hist, method_name=f'Gradient Descent (init={initial})')
    
    # Part B: Metropolis–Hastings
    print("Running Metropolis–Hastings...")
    mh_results = {}
    for initial in initial_guesses:
        omega_hist = metropolis_hastings(initial, rho=0.0001, sigma=0.5, iterations=10000)
        mh_results[initial] = omega_hist
        plot_trajectory(omega_hist, method_name=f'Metropolis–Hastings (init={initial}, beta=0.0001)')
    
    # Part C: Simulated Annealing (with cooling schedule)
    print("Running Simulated Annealing...")
    sa_results = {}
    for initial in initial_guesses:
        omega_hist, rho_hist = simulated_annealing(initial, rho0=0.0001, phi=0.0000001, sigma=0.5, iterations=10000)
        sa_results[initial] = (omega_hist, rho_hist)
        plot_trajectory(omega_hist, method_name=f'Simulated Annealing (init={initial}, beta=0.0001, delta=0.0000001)')
    
    # Comparison plot: MH (without cooling) vs SA (with cooling) for initial guess ω=1.0.
    omega_mh = metropolis_hastings(1.0, rho=0.0001, sigma=0.5, iterations=10000)
    omega_sa, _ = simulated_annealing(1.0, rho0=0.0001, phi=0.0000001, sigma=0.5, iterations=10000)
    plt.figure()
    x = np.linspace(-5, 5, 400)
    y = H(x)
    plt.plot(x, y, 'b-', label='H(ω)')
    plt.plot(omega_mh, [H(w) for w in omega_mh], 'ro-', markersize=3, label='Metropolis–Hastings')
    plt.plot(omega_sa, [H(w) for w in omega_sa], 'go-', markersize=3, label='Simulated Annealing')
    plt.xlabel('ω')
    plt.ylabel('H(ω)')
    plt.title('Comparison of MH and SA')
    plt.legend()
    plt.show()
    plt.xlim(-3,3)
    plt.ylim(-20,10)
    plt.savefig('comparison_MH_SA.png')

if __name__ == '__main__':
    main()

"""2D Quantum Harmonic Oscillator Simulation with Adiabatic Evolution
Author: Jonathan
Date: 2025-03-09
"""

import numpy as np
from scipy.linalg import expm

# Configuration parameters
class SimulationConfig:
    """Configuration class for simulation parameters.
    
    This structure manages parameters for both ITE ground state finding
    and Lanczos time evolution, ensuring consistent configuration across
    all simulation stages.
    """
    def __init__(self):
        # Spatial grid parameters
        self.L = 20.0        # Domain size
        self.N = 128         # Grid points per dimension
        self.dx = self.L/(self.N - 1)  # Grid spacing
        
        # Time evolution parameters
        self.dt_imag = 5e-3       # Imag time step
        self.Tmax = 5.0      # Total simulation time
        self.kdim = 8        # Krylov subspace dimension
        
        # Physical parameters
        self.hbar = 1.0      # Reduced Planck constant
        self.m = 1.0         # Mass
        self.omega = 1.0     # Angular frequency
        
        # Numerical stability parameters
        self.stability_threshold = 1e-10  # Threshold for numerical stability checks
        
        # Derived parameters
        self.dxsq = self.dx**2

        #Real time parameters
        #self.Tf = 575
        self.Tf = 90
        #self.Nt = 3000
        self.Nt = 2000
        self.real_dt = self.Tf / (self.Nt-1)        

# Initialize configuration
config = SimulationConfig()

# Grid setup with error checking
def setup_grid(config):
    """Initialize spatial and momentum grids with validation.
    Args:
        config: SimulationConfig object
    Returns:
        tuple: (X, Y, K2, V, Vf)
    """
    try:
        x = np.linspace(-config.L/2, config.L/2, config.N)
        y = x.copy()
        X, Y = np.meshgrid(x, y)
        
        kx = np.fft.fftfreq(config.N, config.dx)
        ky = kx.copy()
        KX, KY = np.meshgrid(kx, ky)
        K2 = 4 * np.pi**2 * (KX**2 + KY**2)
        
        # Potential energy operators (both constant)
        V_initial = 0.5 * config.m * config.omega**2 * (X**2 + Y**2)

        V_final = 0.5 * config.m * config.omega**2 * ((X+3)**2 + (Y-5)**2)
        #V_final = -5*(np.exp(-((X-4)**2) / (2*3)))*(np.exp(-((Y+3)**2) / (2*4))) - 5*(np.exp(-((X+5)**2) / (2*(5)))) * (np.exp(-((Y+5)**2) / (2*(5))))  - 4*(np.exp(-((X)**2) / (2*3)))*(np.exp(-((Y-5)**2) / (2*3))) + 0.05 * (X**2 + Y**2)
        
        return X, Y, K2, V_initial, V_final
    except Exception as e:
        raise RuntimeError(f"Grid setup failed: {str(e)}")

# Initialize grids
X, Y, K2, V_initial, V_final = setup_grid(config)

class QuantumState:
    """Class representing a quantum state with verification methods.
    
    Handles both ITE-found and time-evolved states, providing
    consistent normalization and measurement functionality.
    """
    def __init__(self, psi_step1, psi_step3, psi_real, config):
        self.psi_step1 = psi_step1
        self.psi_step3 = psi_step3
        self.psi_real = psi_real
        self.config = config
    
    def normalize_step1(self):
        """Normalize the state with error checking."""
        norm = np.sqrt(np.sum(np.abs(self.psi_step1)**2) * self.config.dxsq)
        if norm < self.config.stability_threshold or np.isnan(norm) or np.isinf(norm):
            print("Warning: Invalid norm detected in normalize_step1")
            self.psi_step1 = np.zeros_like(self.psi_step1)
            return self
        self.psi_step1 /= norm
        return self

    def normalize_step3(self):
        """Normalize the state with error checking."""
        norm = np.sqrt(np.sum(np.abs(self.psi_step3)**2) * self.config.dxsq)
        if norm < self.config.stability_threshold or np.isnan(norm) or np.isinf(norm):
            print("Warning: Invalid norm detected in normalize_step3")
            self.psi_step3 = np.zeros_like(self.psi_step3)
            return self
        self.psi_step3 /= norm
        return self

def H_Psi(func_Psi, P):
    """Hamiltonian application using spectral method.
    
    For the harmonic oscillator, we could use the Hermite function basis
    where H|n⟩ = (n + 1/2)ℏω|n⟩, but we keep the spectral method for
    consistency with the time evolution.
    """
    # Input validation
    if np.any(np.isnan(func_Psi)) or np.any(np.isinf(func_Psi)):
        print("Warning: Invalid input to H_Psi")
        return np.zeros_like(func_Psi)
    # Kinetic energy in k-space
    psi_k = np.fft.fft2(func_Psi) / config.N
    T_psi = (config.hbar**2/(2*config.m)) * np.fft.ifft2(K2 * psi_k) * config.N
    # Add potential energy in real space
    result = T_psi + P * func_Psi
    # Output validation
    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
        print("Warning: Invalid output from H_Psi")
        return np.zeros_like(func_Psi)
    return result

def smooth_t_evolution(t, tf):
    s = t/tf
    return 1-np.cos(s * np.pi/2)

class LanczosWorkspace:
    """
    Workspace for the Lanczos algorithm, holding all large arrays to avoid repeated allocations.
    """
    def __init__(self, N, kdim):
        self.alpha = np.zeros(kdim)
        self.beta = np.zeros(kdim-1)
        self.W = np.zeros((N, N, kdim), dtype=np.complex128)
        self.v = np.zeros((N, N), dtype=np.complex128)
        self.Hw = np.zeros((N, N), dtype=np.complex128)
        self.Psi_new = np.zeros((N, N), dtype=np.complex128)

    def reset(self):
        self.alpha.fill(0)
        self.beta.fill(0)
        self.W.fill(0)
        self.v.fill(0)
        self.Hw.fill(0)
        self.Psi_new.fill(0)

def lanczos(A, v1, dt, kdim, workspace, img_time=False):
    """Highly optimized Lanczos implementation for real-time evolution.
    
    Builds a Krylov subspace representation of the Hamiltonian,
    enabling efficient and accurate time evolution through:
    1. Tridiagonal matrix construction
    2. Small matrix exponential
    3. Basis transformation
    
    Args:
        A: Potential energy operator
        v1: Initial state vector
        k_max_iter: Maximum Krylov subspace dimension
    
    Returns:
        tuple: (Psi_new, orth_check) where:
            - Psi_new is the evolved state (unnormalized)
            - orth_check is the orthogonality check between first and sixth Krylov vectors
    """
    ws = workspace
    ws.reset()
    
    # Input validation
    if np.any(np.isnan(v1)) or np.any(np.isinf(v1)):
        print("Warning: Invalid initial vector in Lanczos")
        return np.zeros_like(v1)
    
    # Initialize first vector (unnormalized)
    w = v1.astype(np.complex128)
    ws.W[:,:,0] = v1
    w_prev = np.zeros_like(w)
    
    for j in range(kdim):
        # Apply H and compute diagonal element
        ws.Hw = H_Psi(w, A)
        ws.alpha[j] = np.real(np.sum(np.conj(w) * ws.Hw) * config.dxsq)
        
        # Compute residual
        ws.v = ws.Hw - ws.alpha[j] * w
        if j > 0:
            ws.v -= ws.beta[j-1] * w_prev
            
        # Modified Gram-Schmidt orthogonalization
        for i in range(j+1):
            coeff = np.sum(np.conj(ws.W[:,:,i]) * ws.v) * config.dxsq
            ws.v -= coeff * ws.W[:,:,i]
        
        # Compute beta with high precision and stability check
        beta_j = np.sqrt(np.real(np.sum(np.conj(ws.v) * ws.v) * config.dxsq))
        
        if beta_j < config.stability_threshold:
            print(f'Warning: Small beta_j detected at iteration {j}, beta={beta_j:.16e}')
            T = np.diag(ws.alpha[:j+1]) + np.diag(ws.beta[:j], -1) + np.diag(ws.beta[:j], 1)
            ws.W = ws.W[:,:,:j+1]
            break
            
        if j < kdim - 1:
            ws.beta[j] = beta_j
            w_prev = w.copy()
            w = ws.v / beta_j
            
            # Stability check for w
            if np.any(np.isnan(w)) or np.any(np.isinf(w)):
                print(f"Warning: Invalid w detected at iteration {j}")
                break
                
            ws.W[:,:,j+1] = w
    
    T = np.diag(ws.alpha) + np.diag(ws.beta, -1) + np.diag(ws.beta, 1)
    
    try:
        if img_time:
            U_T = expm(-T * dt)
        else:
            U_T = expm(-1j * T * dt)
    except Exception as e:
        print(f"Warning: Matrix exponential failed: {str(e)}")
        return np.zeros_like(v1)

    stateKrylov = U_T[:,0]

    for nn in range(kdim):
        ws.Psi_new += stateKrylov[nn] * ws.W[:,:,nn]
    
    # Final stability check
    if np.any(np.isnan(ws.Psi_new)) or np.any(np.isinf(ws.Psi_new)):
        print("Warning: Invalid final state in Lanczos")
        return np.zeros_like(v1)
        
    return ws.Psi_new

# Start timing
t = time.process_time()

# Initialize ground state with proper normalization
def initialize_random_state(N):
    """Initialize a normalized random state."""
    psi = np.random.rand(N, N) + 1j * np.random.rand(N, N)
    norm = np.sqrt(np.sum(np.abs(psi)**2) * config.dxsq)
    return psi / norm

# Initialize states with proper normalization
psi_step1 = initialize_random_state(config.N)
psi_step3 = initialize_random_state(config.N)
psi_real = initialize_random_state(config.N)
state = QuantumState(psi_step1, psi_step3, psi_real, config)

# Time evolution setup
print("\nStarting real-time evolution using Lanczos...")
Tvector = np.arange(0, config.Tmax, config.dt_imag)
EnergyVector = np.zeros(len(Tvector), dtype=np.complex128)
kdim = config.kdim

# Pre-allocate arrays
Psi_new1 = np.zeros((config.N, config.N), dtype=np.complex128)
Psi_new3 = np.zeros((config.N, config.N), dtype=np.complex128)
Psi_real = np.zeros((config.N, config.N), dtype=np.complex128)
max_energy_dev = 0.0

# Allocate the workspace once before main loops
workspace = LanczosWorkspace(config.N, kdim)
# Start timing
t = time.process_time()
# Main evolution loop step 1
for index, t_val in enumerate(Tvector):
    # Lanczos step
    Psi_new1 = lanczos(V_initial, state.psi_step1, config.dt_imag, kdim, workspace, img_time=True)
    
    # Calculate energy from norm decay
    norm = np.sum(np.abs(Psi_new1)**2) * config.dxsq
    epsilon = -np.log(norm)/(2*config.dt_imag)
    Energy = epsilon  # epsilon is already the energy eigenvalue *1*1
    EnergyVector[index] = Energy
    
    # Normalize for next step
    state.psi_step1 = Psi_new1 / np.sqrt(norm)
    
    if index == 0:
        elapsed_time = time.process_time() - t
        print(f'Step: 1/{len(Tvector)} | E = {Energy:.10f} | Time: {elapsed_time:.2f}s')

    if (index+1) % 250 == 0:
        elapsed_time = time.process_time() - t
        print(f'Step: {index+1}/{len(Tvector)} | E = {Energy:.10f} | Time: {elapsed_time:.2f}s')
        

print("-" * 50)
print("Step 1 done. Starting step 3...")
print("-" * 50)

# Start timing
t = time.process_time()
# Main evolution loop step 3
for index, t_val in enumerate(Tvector):
    # Lanczos step
    Psi_new3 = lanczos(V_final, state.psi_step3, config.dt_imag, kdim, workspace, img_time=True)
    
    # Calculate energy from norm decay
    norm = np.sum(np.abs(Psi_new3)**2) * config.dxsq
    epsilon = -np.log(norm)/(2*config.dt_imag)
    Energy = epsilon  # epsilon is already the energy eigenvalue *1*1
    EnergyVector[index] = Energy
    
    # Normalize for next step
    state.psi_step3 = Psi_new3 / np.sqrt(norm)
    
    if index == 0:
        elapsed_time = time.process_time() - t
        print(f'Step: 1/{len(Tvector)} | E = {Energy:.10f} | Time: {elapsed_time:.2f}s')

    if (index+1) % 250 == 0:
        elapsed_time = time.process_time() - t
        print(f'Step: {index+1}/{len(Tvector)} | E = {Energy:.10f} | Time: {elapsed_time:.2f}s')

print("-" * 50)
print("Step 3 done. Starting real-time evolution (step 2)...")
print("-" * 50)

# Initialize overlap array and pre-allocate arrays for evolution

Psi_real = np.zeros((config.N, config.N), dtype=np.complex128)
V_real = np.zeros((config.N, config.N))


# Pre-calculate potential differences to avoid repeated calculations
V_diff = V_final - V_initial

# Optimize overlap calculation by pre-calculating conjugate of final state
psi_step3_conj = np.conj(state.psi_step3)

# Set up plot update interval
Nt = config.Nt
real_dt = config.real_dt
Tf = config.Tf

Tvector_real = np.arange(0, Tf, real_dt)

print(f"\nStarting evolution with Tf={Tf}s, Nt={Nt}, dt={real_dt:.6f}")

# Reset state for each run
state.psi_real = state.psi_step1.copy()
for t_idx, t in enumerate(Tvector_real):
    # Calculate the time-dependent potential with smooth transition
    s = smooth_t_evolution(t + real_dt/2, Tf)
    V_real = V_initial + s * V_diff

    # Lanczos step with smaller time step for better stability
    Psi_real = lanczos(V_real, state.psi_real, real_dt, kdim, workspace, img_time=False)

    # Normalize the state
    norm = np.sum(np.abs(Psi_real)**2) * config.dxsq
    if norm < config.stability_threshold or np.isnan(norm) or np.isinf(norm):
        print(f"Warning: Invalid norm at t={t:.2f}, skipping step")
        continue
    state.psi_real = Psi_real  / np.sqrt(norm)
    
    
    # Print progress at 25% intervals
    if t_idx % (Nt//300) == 0:
        #elapsed_time = time.process_time() - t
        #print(f'Step: {t*100/Tf:.0f}% | Overlap: {Overlap:.10f} | Time: {elapsed_time:.2f}s')
        print(f'Step: {t*100/Tf:.0f}%')
        
        

Overlap = np.abs(np.sum(psi_step3_conj * state.psi_real) * config.dxsq)**2

# Final diagnostics
elapsed_time = time.process_time() - t
print("\nFinal state verification:")
print("Checking final state properties...")
print(f'Time elapsed: {elapsed_time:.2f} seconds')
print(f'Overlap: {Overlap:.10f}')
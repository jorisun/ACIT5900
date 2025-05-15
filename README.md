# Quantum Annealing in 2D Harmonic Oscillators For ACIT5900

High-precision simulation of quantum annealing using Imaginary Time Evolution (ITE) and the Lanczos algorithm, applied to dynamic evolution between two 2D potential landscapes.

---

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Theoretical Background](#theoretical-background)
- [Implementation Details](#implementation-details)
- [Key Features](#key_features)
- [Simulation Workflow](#simulation_workflow)
- [Numerical Parameters](#numerical_parameters)
- [Visualization](#visualization)

---

## Overview

This module implements a high-precision quantum simulation of a 2D harmonic oscillator,
demonstrating adiabatic evolution between two different potential configurations.
The simulation combines two powerful numerical methods:
1. Imaginary Time Evolution (ITE) for finding ground states
2. Lanczos algorithm for dynamic evolution

---

## Installation

Requires Python 3.8+

1. Clone the Repository:
git clone https://github.com/jorisun/ACIT5900.git
cd ACIT5900

2. Install Dependencies:
pip install -r requirements.txt

If requirements.txt is not available, do:
pip install numpy scipy

3. Run the Simulation:
python Quantum_Annealing_simulation_2d.py

---

## Theoretical Background

1. Quantum Harmonic Oscillator:
   - Hamiltonian: H = -ℏ²/(2m)∇² + (1/2)mω²r²
   - Ground state energy: E₀ = ℏω
   - Ground state wavefunction: ψ₀(x,y) = (mω/πℏ)^(1/2) * exp(-mωr²/2ℏ)
   - Energy spectrum: E_n = (n + 1/2)ℏω
2. Adiabatic Evolution:
   - Time-dependent Hamiltonian: H(t) = (1-s(t))H_initial + s(t)H_final
   - Smooth transition function: s(t) = 35(t/Tf)⁴ - 84(t/Tf)⁵ + 70(t/Tf)⁶ - 20(t/Tf)⁷
   - Adiabatic theorem: System remains in instantaneous ground state if evolution is slow enough
   - Overlap measure: |⟨ψ_final|ψ(t)⟩|² quantifies adiabaticity
3. Numerical Methods:
   a) Imaginary Time Evolution:
      - Projects arbitrary state onto ground-state
      - Evolution operator: exp(-τH) where τ is imaginary time
      - Energy extracted from norm decay rate
   b) Lanczos Algorithm:
      - Builds Krylov subspace: {v, Hv, H²v, ...}
      - Tridiagonalizes Hamiltonian in this subspace
      - Computes matrix exponential efficiently
      - Preserves unitarity in real-time evolution

---

## Implementation Details

1. Grid Setup:
   - Uniform grid in real space: [-L/2, L/2]x[-L/2, L/2]
   - FFT-based spectral method for kinetic energy
   - Momentum space grid for spectral accuracy
2. Potential Configuration:
   - Initial: V_initial = (1/2)mω²(x² + y²)
   - Final: V_final = (1/2)mω²((x+3)² + (y-5)²)
   - Shifted potential tests adiabatic evolution
3. Numerical Stability:
   - Modified Gram-Schmidt orthogonalization
   - Stability threshold for numerical operations
   - Norm preservation checks
   - Error tracking and reporting
4. Performance Optimization:
   - Pre-allocated arrays for efficiency
   - FFT-based spectral methods
   - Adaptive time stepping

---

## Key Features

- High-precision ground state finding via ITE
- Spectral accuracy using FFT methods
- Efficient time evolution using Lanczos
- Comprehensive error tracking
- Analytical verification against exact solutions
- Adaptive numerical stability

---

## Simulation Workflow

1. Initialize random state with proper normalization
2. Find initial ground state using ITE
3. Find final ground state using ITE
4. Perform adiabatic evolution between states
5. Monitor overlap and energy conservation
6. Analyze results and verify adiabaticity

---

## Numerical Parameters

- Domain size: L=20
- Grid points per dimension: N=128
- Grid size: {N}x{N} points
- Domain: [-L/2, L/2]x[-L/2, L/2], L={L}
- Spatial resolution: dx={L/N}
- Imaginary time step: dt_imag=1e-3
- Krylov subspace dimension: kdim=8
- Stability threshold: 1e-10

- Reduced Plank constant: hbar =1
- Particle mass: m=1
- Angular frequency: omega=1

---

## Visualization

Quantum Annealing into Simple Harmonic Oscillator Potential Animation:
https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExcnpqZ3hydWV5aXUyYmZib3Bqd2RiMXkzZmF4dWtpOWJ1N2l0NTdocCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Ge4lCznXm9gdu9Fxmr/giphy.gif 

Quantum Annealing into Complex Gaussian Multi-Well Potential Animation:
https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExYjg0ZWdkbXhvcWY3ajNvaWVjbW8zZ2J2MHdmMm1tN3NpYjR0NzRsYSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/OKDVvq8uTpS9kxNO7y/giphy.gif


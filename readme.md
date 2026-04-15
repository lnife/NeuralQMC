# Neural Variational Monte Carlo (Helium)

A minimal implementation of **Variational Monte Carlo (VMC)** using a neural wavefunction to approximate the ground state energy of the Helium atom.

This project replaces analytical wavefunctions with a **learned ansatz**, while retaining the physical structure required for fermionic systems.

---

## Overview

The wavefunction is modeled as:

ψ(x) = exp( log|det(Slater)| + Jastrow )

- The **Slater determinant** enforces antisymmetry

- The **Jastrow factor** encodes electron correlation

- A neural network parameterizes the orbital structure

Sampling is performed using Metropolis-Hastings.  
Optimization is driven by the variational principle.

---

## Project Structure

```id="q1x9az"
.
├── main.py           # Entry point
├── train.py          # Training loop
├── wavefunction.py   # Combines Slater + Jastrow
├── slater.py         # Neural Slater determinant
├── jastrow.py        # Correlation factor
├── sampler.py        # Metropolis-Hastings sampling
├── hamiltonian.py    # Local energy computation
├── distances.py      # Electron distance calculations
├── config.py         # Hyperparameters
```

---

## Motivation

This project was developed as a continuation of direct, first-principles exploration of electronic structure methods.

Instead of:

- Closed-form hydrogenic solutions

- Deterministic orbital evaluation

we move to:

- Learned wavefunctions

- Stochastic sampling

- Energy minimization via gradients

The goal is not accuracy alone.

The goal is understanding:

- How antisymmetry is enforced in learned systems

- How correlation emerges from simple parametrizations

- How Monte Carlo sampling interacts with optimization

- How automatic differentiation replaces analytic Laplacians

- Where numerical instability appears in log-space formulations

This is a study of **variational quantum mechanics as an algorithm**, not a black-box method.

---

## Architecture

### Wavefunction (`wavefunction.py`)

Combines:

- Slater determinant (antisymmetric structure)

- Jastrow factor (correlation)

Outputs log|ψ| for numerical stability.

---

### Slater Determinant (`slater.py`)

- Neural network maps electron coordinates → orbital values

- Constructs a 2×2 determinant (Helium system)

- Log-determinant used to avoid numerical underflow

---

### Jastrow Factor (`jastrow.py`)

- Pade form correlation function

- Learnable parameters

- Encodes short-range electron-electron behavior

---

### Sampling (`sampler.py`)

- Metropolis-Hastings updates

- Log-probability ratio for stability

- No drift term (pure random walk)

---

### Local Energy (`hamiltonian.py`)

Computed via automatic differentiation:

- Gradient of logψ

- Laplacian via second derivatives

Kinetic term:  
-½ (∇² logψ + |∇ logψ|²)

Potential:

- Electron-nucleus attraction

- Electron-electron repulsion

---

### Training (`train.py`)

- Energy expectation minimization

- Variance-reduced gradient estimator

- Adam optimizer with step decay

---

### Entry Point (`main.py`)

- Initializes walkers

- Performs thermalization

- Starts optimization loop

---

## Numerical Strategy

### Representation

- Log-space wavefunction to prevent overflow

- Determinant computed explicitly (2-electron system)

---

### Sampling

- Walkers represent electron configurations

- Distribution converges to |ψ|²

---

### Optimization

Loss:

L = ⟨ (E_L - ⟨E_L⟩) logψ ⟩

This avoids direct differentiation of the normalization constant.

---

## Running

```bash
python main.py
```

Output:

```id="p8d2xw"
step N  E = ...  var = ...
```

Reference:

```id="z7v4rt"
Helium ground state ≈ -2.903 Hartree
```

---

## Limitations

- Fixed to two electrons

- No spin-explicit formalism

- No importance sampling (no drift term)

- Single-determinant ansatz

- CPU-only execution

**Current shortcoming:**

The implementation is **computationally slow**.

This is primarily due to:

- Second-order autodiff for Laplacian computation

- Lack of vectorized or optimized sampling

- No GPU acceleration

- Repeated graph construction during energy evaluation

Performance has not been optimized.  
Clarity and correctness were prioritized over efficiency.

---

## Design Philosophy

- Physics structure is preserved, not replaced

- Neural networks are used where analytical forms are restrictive

- Numerical transparency is prioritized over performance

- No external quantum chemistry libraries

Everything is implemented explicitly to expose the mechanics of VMC.

---

## Future Directions

- Drift-diffusion (importance sampling)

- Multi-determinant expansions

- GPU acceleration

- More efficient Laplacian computation

- Equivariant neural architectures

- Scaling beyond two-electron systems

---

## References

1. Slater determinant formalism  
   J. C. Slater, _The Theory of Complex Spectra_, Phys. Rev. **34**, 1293–1322 (1929).  
   DOI: 10.1103/PhysRev.34.1293

2. Hartree product / independent-particle approximation  
   D. R. Hartree, _The Wave Mechanics of an Atom with a Non-Coulomb Central Field. Part I. Theory and Methods_,  
   Math. Proc. Cambridge Philos. Soc. **24**, 89–110 (1928).

3. Electron-electron correlation (Jastrow ansatz origin)  
   R. Jastrow, _Many-Body Problem with Strong Forces_, Phys. Rev. **98**, 1479–1484 (1955).

4. Cusp condition for Coulomb singularities  
   T. Kato, _On the Eigenfunctions of Many-Particle Systems in Quantum Mechanics_,  
   Commun. Pure Appl. Math. **10**, 151–177 (1957).

5. Helium Hamiltonian / two-electron Coulomb problem  
   E. A. Hylleraas, _Neue Berechnung der Energie des Heliums im Grundzustande_,  
   Z. Phys. **54**, 347–366 (1929).

6. Padé Jastrow correlation form in modern electronic QMC  
   C. J. Umrigar, M. P. Nightingale, K. J. Runge,  
   _A Diffusion Monte Carlo Algorithm with Very Small Time-Step Errors_,  
   J. Chem. Phys. **99**, 2865 (1993).

7. Explicit correlated helium wavefunction product ansatz  
   R. S. Chauhan and M. K. Harbola,  
   _Highly Accurate Wavefunctions for Two-Electron Systems Using Two Parameters_,  
   arXiv:1506.00912 (2015).  
   DOI: 10.48550/arXiv.1506.00912

8. Slater-Jastrow trial wavefunction in modern neural QMC  
   D. Pfau, J. S. Spencer, A. G. D. G. Matthews, W. M. C. Foulkes,  
   _Ab initio solution of the many-electron Schrödinger equation with deep neural networks_,  
   Phys. Rev. Research **2**, 033429 (2020).  
   DOI: 10.1103/PhysRevResearch.2.033429

9. Variational / diffusion Monte Carlo foundational review  
   D. M. Ceperley and B. J. Alder,  
   _Quantum Monte Carlo_, Science **231**, 555–560 (1986).

---

## Author

Lnifelias Stargarden  
Real name: Bhaskar Malviya

Computational Chemistry | Quantum Chemistry | Scientific Programming

---

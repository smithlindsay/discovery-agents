# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project builds simulated physical worlds where RL-trained LLM agents discover unknown laws of physics through experimentation. Agents observe noisy particle trajectories, design experiments, and propose governing equations. The simulator generates diverse worlds by randomizing field equations, particle-field couplings, and symmetry structure.

The core equation family:
```
∂ⁿφ/∂tⁿ = L[φ] + N[φ] + S(particles)
```
where `n ∈ {0, 1, 2}`, `L` is a linear spatial operator, `N` contains nonlinear terms, and `S` couples particles to the field.

## Repository Structure

- **`PhysicsSchool/`** — Main physics simulation engine (active development). See `PhysicsSchool/CLAUDE.md` for detailed spec.
- **`ScienceAgent/`** — Planned RL/LLM agent component (currently empty).

## Development Commands

```bash
# Install the package (from PhysicsSchool/)
pip install -e .

# Run all tests
pytest PhysicsSchool/tests/

# Run a single test file
pytest PhysicsSchool/tests/test_forces.py

# Run a single test
pytest PhysicsSchool/tests/test_forces.py::test_function_name -x -q
```

## HPC (Flatiron "rusty" cluster)

```bash
# Interactive GPU session for testing
salloc -p gpu --gres=gpu --mem=128G --time=2:00:00
source /mnt/home/ccuesta/ceph/environments/YOUR_ENV/bin/activate
pytest PhysicsSchool/tests/ --fast -x -q

# Batch training job
sbatch scripts/train.sbatch
squeue -u ccuesta
```

Data: `/mnt/home/ccuesta/ceph/physics_school_data`
Environments: `/mnt/home/ccuesta/ceph/environments`

## Architecture: PhysicsSchool

**Stack:** JAX (all simulation code), pytest, numpy (initialization only)

**Key classes/files:**
- `physchool/worlds/field_sampler.py` — `FieldSampler` class: the single entry point for all simulation. Handles n=0 (constraint/Poisson via FFT), n=1 (diffusion, explicit Euler), n=2 (wave, symplectic Euler).
- `physchool/worlds/utils.py` — `cic_paint()` and `cic_read()`: Cloud-In-Cell particle-mesh interpolation. Do not modify.

**Planned new modules** (not yet created):
- `worlds/operators.py` — Linear and nonlinear operator implementations
- `worlds/multi_field.py` — Multi-field extension
- `worlds/sampling.py` — `sample_world()` and `SymmetryConfig`
- `worlds/validation.py` — Validation test suite

**Critical implementation constraints:**
- All hot-path code must be JAX/JIT-compilable. `step()` must be a pure function (state in → new state out, no side effects).
- Use JAX autodiff for field gradients — **not** finite differences (`np.roll`). The existing code uses finite differences and must be replaced.
- Use `jax.lax.scan` for multi-step rollouts to compile full trajectories into a single XLA computation.
- Performance target: full step for <100 particles on 64×64 grid → <10ms GPU, <50ms CPU.

**Supported linear operators** (via Fourier-space multiplication):
- `laplacian` (-k²), `fractional_laplacian` (-k^2α), `helmholtz` (-(k²+m²)), `screening`/`yukawa` (-(k²+1/λ²)), `identity`

**Operator specification format:**
```python
operators = [
    {"type": "laplacian", "params": {"strength": 1.0}},
    {"type": "advection", "params": {"strength": 0.5}},  # nonlinear (planned)
]
```

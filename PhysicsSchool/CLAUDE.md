# CLAUDE.md — Physics World Simulator

## Project Context

We are building simulated physical worlds where an RL-trained LLM agent must discover unknown laws of physics through experimentation. The agent observes noisy particle trajectories, designs experiments (sets initial conditions), and proposes the governing equations. The simulator generates diverse worlds by randomizing the field equations, particle-field couplings, and symmetry structure, forcing the agent to do genuine scientific reasoning rather than pattern matching against known physics.

The core equation family is:

    ∂ⁿφ/∂tⁿ = L[φ] + N[φ] + S(particles)

where n ∈ {0, 1, 2} is the temporal order, L is a linear spatial operator, N contains nonlinear terms, and S couples particles to the field. Particles feel forces from the field and move according to Newton's second law.

The existing code (`physchool/worlds/field_sampler.py`) implements the linear case with a single scalar field and has not been tested properly. We want to extend it along four axes: extensive testing and reformating, nonlinear field dynamics, multi-field coupling, and systematic symmetry variation.

## Technical Requirements

### Language and Framework

- All simulation code must be in **JAX**. Use `jax.numpy` instead of `numpy` for any array operations that touch the simulation loop.
- Use `jax.jit` for the hot path (field solves, force computation, time stepping). The `step()` function should be JIT-compilable as a whole.
- Use the existing `cic_paint` and `cic_read` from `physchool.worlds.utils` for particle-mesh interpolation. Do not rewrite these.
- Gradient computation on the field should use **JAX autodiff** (`jax.grad` or `jax.jacfwd`), not finite differences. This is both more accurate and naturally composable with JIT. The current code uses finite difference `np.roll` for gradients — replace this.
- Target performance: a full step (field solve + force computation + particle update) for <100 particles on a 64×64 grid should take <10ms on a single GPU, or <50ms on CPU. This matters because RL training will call `step()` many times.

### 0. Reformat and test the current implementation
Start by checking the current implementation, reformat the code with simplicity in mind, and make sure to test the code with known analytical solutions

### 1. Nonlinear Field Terms

Add nonlinear terms to the n=1 and n=2 time-stepping branches. These are evaluated on the grid at each timestep and added to the RHS.

Implement at least the following nonlinear operators:
- **Advection/Burgers**: `strength * φ * ∇φ` (produces shocks)
- **Reaction (quadratic)**: `strength * φ²` (breaks φ → -φ parity)
- **Reaction (cubic)**: `strength * φ³` (preserves φ → -φ parity)
- **Square gradient**: `strength * |∇φ|²` (nonlinear energy-type term)
- **KPP/logistic**: `strength * φ * (1 - φ)` (saturation dynamics)

These should be specified in the same operator list format as the linear operators, e.g.:
```python
operators = [
    {"type": "laplacian", "params": {"strength": 1.0}},
    {"type": "advection", "params": {"strength": 0.5}},
    {"type": "cubic", "params": {"strength": -0.1}},
]
```

**Important**: nonlinear terms can cause numerical instability. For the n=1 and n=2 branches:
- Use a CFL-aware timestep check: warn or error if `dt > C * dx / max_velocity` where `max_velocity` is estimated from the field values.
- Consider implementing a simple exponential integrator or split-step method for stiff problems (e.g. when the Laplacian coefficient is large relative to the nonlinear terms). At minimum, document which operator combinations are expected to be stable.

**The n=0 constraint equation remains linear.** Nonlinear constraint equations (like MOND) would require iterative solvers — defer this to a later phase. For now, n=0 worlds get their complexity from exotic linear operators (fractional Laplacian, screening) and from particle heterogeneity.

### 2. Multi-Field Systems

Extend the simulator to support multiple interacting fields. The state becomes a list of fields `[φ₁, φ₂, ...]`, each with its own temporal order, operators, and coupling to particles.

The coupling between fields should be specified as a matrix of interaction terms:
```python
field_couplings = [
    {"source_field": 0, "target_field": 1, "type": "linear", "strength": 0.5},
    {"source_field": 1, "target_field": 0, "type": "gradient_coupling", "strength": -0.3},
]
```

Where coupling types include:
- **linear**: `strength * φ_source` added to the RHS of the target field equation
- **gradient_coupling**: `strength * ∇φ_source · ∇φ_target` (energy-type interaction)
- **multiplicative**: `strength * φ_source * φ_target` (nonlinear cross-coupling)

Each field independently couples to particles through its own `source_coupling` and `force_coupling`. The total force on a particle is the sum of forces from all fields.

Start with support for 2-3 fields. The data structure should generalize to N fields but performance optimization beyond 3 is not a priority.

### 3. Symmetry Tracking

Each generated world should carry metadata describing its symmetry properties. This is not enforced by the simulator — it is *derived* from the operator and coupling choices, and stored for use in reward computation.

Track at minimum:
- **Translational symmetry**: always true for periodic domains
- **Rotational symmetry**: true if all spatial operators are isotropic (no anisotropic Laplacian, no preferred direction in couplings)
- **Time-reversal symmetry**: true for n=0 and n=2 without damping; false for n=1 or if explicit damping terms are present
- **Parity (φ → -φ)**: true if all nonlinear terms have odd powers of φ; false if even powers are present
- **Charge conjugation**: true if swapping the sign of all particle sources gives identical dynamics

Store this as a dictionary on the `FieldSampler` instance so it can be queried by the reward function.

### 4. Validation and Correctness

Every new feature must be validated. This is critical — if the simulator is wrong, the agent learns wrong physics.

**For the Poisson/constraint solver (n=0):**
- Compare against the analytic Fourier Green's function for a single point source. The Dedalus spectral solver and the numpy FFT solver should agree to machine precision (relative error < 1e-10).
- For fractional Laplacian and screening operators, verify the correct k-space scaling: a point source in a Yukawa field should produce φ(r) ∝ K₀(r/λ) (modified Bessel function) at distances much less than the box size.

**For time-dependent fields (n=1, n=2):**
- Diffusion equation (n=1, Laplacian only): a Gaussian initial condition should spread as σ²(t) = σ₀² + 2αt where α is the diffusion coefficient. Verify this quantitatively.
- Wave equation (n=2, Laplacian only): a sinusoidal initial condition should propagate at speed c = √α. Measure the phase velocity and compare.
- For nonlinear terms: Burgers' equation with known initial conditions has well-characterized shock formation times. Verify the shock forms at the expected time.

**For particle dynamics:**
- **Mass conservation**: CIC deposit then integrate over grid should return total particle mass to machine precision.
- **Momentum conservation**: total momentum should be conserved to ~machine precision for periodic domains with gradient force law (Newton's 3rd law via CIC symmetry).
- **Energy conservation**: for n=0 and n=2 without dissipation, total energy (kinetic + potential) should oscillate without secular drift over many timesteps. Use this as a regression test — if energy drifts, something is wrong.
- **Resolution convergence**: run the same initial conditions at 32², 64², 128² and verify that particle trajectories converge. The error should decrease with resolution.

**For multi-field systems:**
- Two uncoupled fields should give identical results to two independent single-field simulations.
- A conserved quantity (e.g. from a U(1) symmetry in field space) should remain constant to integrator accuracy.

Write validation tests as standalone functions that can be run with a `--validate` flag. Print PASS/FAIL with quantitative error metrics. These tests should run in under 2 minutes total.

### 5. World Sampling

Implement a `sample_world()` function that generates a random `FieldSampler` configuration. This is the distribution of training environments for RL.

The sampler should:
- Choose temporal order n ∈ {0, 1, 2} with configurable probabilities
- Choose 1-3 fields, each with random operators and couplings
- Randomize particle properties (inertia, source, force) from a mixture of 1-3 discrete types, so the agent must discover particle species
- Randomize which nonlinear terms are present and their strengths
- Ensure the resulting system is numerically stable (reject configurations where a test run of 100 steps produces NaN or energies that blow up)
- Record the symmetry metadata for the sampled world
- Be reproducible given a random seed

### 6. Efficiency Notes

- The field solve for n=0 (FFT-based) is already fast. Do not over-optimize this.
- The bottleneck for n=1 and n=2 will be the per-step nonlinear term evaluation. Keep these as simple grid operations (pointwise and stencil) that JIT well.
- CIC paint/read involves a scatter/gather over particles, which can be slow in pure JAX for small particle counts due to overhead. If `cic_paint`/`cic_read` from `physchool.worlds.utils` are slow for <100 particles, profile before rewriting — the overhead may be negligible relative to the FFT.
- Avoid Python loops over timesteps where possible. If the full trajectory (many steps) is needed, consider using `jax.lax.scan` to compile the entire rollout into a single XLA computation. This is important for RL training where you need fast rollouts.
- Do NOT use `jax.jit` with side effects or in-place mutation. The `step()` function should be purely functional: take state in, return new state out.

### 7. Code Structure

```
physchool/
  worlds/
    field_sampler.py    — main FieldSampler class (extend this)
    utils.py            — cic_paint, cic_read (do not modify)
    operators.py        — linear and nonlinear operator implementations (new)
    multi_field.py      — multi-field extension (new)
    sampling.py         — sample_world() and SymmetryConfig (new)
    validation.py       — validation test suite (new)
```

Keep `FieldSampler` as the single entry point. Multi-field support should be a generalization of the existing class, not a separate class. A single-field world is just the special case of a multi-field world with one field.

## HPC info


You are on the Flatiron **rusty** cluster.

**Training runs** → submit with `sbatch` (non-blocking, runs in background):
```bash
#!/bin/bash
#SBATCH --job-name=physicsschool
#SBATCH --nodes=1
#SBATCH --mem=128GB
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:a100-sxm4-80gb
#SBATCH --partition=gpu
#SBATCH -o logs/train_%j.out

source /mnt/home/ccuesta/ceph/environments/YOUR_ENV/bin/activate
python scripts/run_train.py --config configs/train.yaml
```
Submit: `sbatch scripts/train.sbatch`. Check: `squeue -u ccuesta`.

**Quick testing** (tests, debugging, short experiments) → use `salloc` for an
interactive GPU session:
```bash
salloc -p gpu --gres=gpu --mem=128G --time=2:00:00
# once allocated:
source /mnt/home/ccuesta/ceph/environments/YOUR_ENV/bin/activate
pytest tests/ --fast -x -q
python -c "import torch; print(torch.cuda.get_device_name())"
```

**Rule of thumb**: if it takes > 10 minutes, use `sbatch`. If it's a quick
test or debugging session, use `salloc`.


All data should be stored at: /mnt/home/ccuesta/ceph/physics_school_data (feel free to create a sym link)
python environments are at /mnt/home/ccuesta/ceph/environments
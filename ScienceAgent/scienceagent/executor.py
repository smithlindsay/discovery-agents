"""
SimulationExecutor: bridges the experiment protocol to FieldSampler.

Protocol mapping
----------------
The discovery protocol exposes two scalar particle properties, p1 and p2.
We map them to FieldSampler parameters as follows:

  p1  → source_coupling of particle 1  (controls field amplitude)
  p2  → particle_inertia of particle 2  (controls how strongly it accelerates)

Both particles have particle_source=1 and particle_force=1 so that the field
operator alone governs the force law the agent must discover.  Particle 1 is
held fixed (infinite effective inertia) by zeroing its acceleration each step.
"""

import json
import numpy as np
import sys
import os

# Make PhysicsSchool importable when running from repo root
_repo_root = os.path.join(os.path.dirname(__file__), "..", "..", "PhysicsSchool")
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from physchool.worlds.field_sampler import FieldSampler


class SimulationExecutor:
    """
    Runs experiments defined by the discovery protocol against a FieldSampler world.

    Args:
        operators: Operator list passed to FieldSampler (the "unknown" physics).
        temporal_order: 0=constraint, 1=diffusion, 2=wave.
        grid_size: Simulation grid resolution.
        domain_size: Physical size of the periodic domain.
        dt: Integration timestep.
    """

    def __init__(
        self,
        operators=None,
        temporal_order=0,
        grid_size=(64, 64),
        domain_size=20.0,
        dt=0.005,
    ):
        self.operators = operators or [{"type": "laplacian", "params": {"strength": 1.0}}]
        self.temporal_order = temporal_order
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.dt = dt

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, experiments: list[dict]) -> list[dict]:
        """
        Run a batch of experiments.

        Args:
            experiments: List of dicts with keys:
                p1, p2, pos2, velocity2, measurement_times
                (duration is inferred as max(measurement_times))

        Returns:
            List of result dicts with keys:
                measurement_times, pos1, pos2, velocity1, velocity2
        """
        return [self._run_one(exp) for exp in experiments]

    def run_json(self, json_str: str) -> str:
        """Parse a JSON string, run experiments, return result as JSON string."""
        experiments = json.loads(json_str)
        results = self.run(experiments)
        return json.dumps(results, indent=2)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_one(self, exp: dict) -> dict:
        p1 = float(exp["p1"])
        p2 = float(exp["p2"])
        pos2 = list(exp["pos2"])
        velocity2 = list(exp["velocity2"])
        measurement_times = sorted(exp["measurement_times"])
        duration = float(exp.get("duration", max(measurement_times)))
        duration = max(duration, 5.0)

        # Particle 1 is fixed at origin; particle 2 is mobile.
        # Positions are placed relative to domain centre so the domain is centred at 0.
        centre = self.domain_size / 2.0
        init_positions = np.array([
            [centre, centre],                          # p1 at origin (domain centre)
            [centre + pos2[0], centre + pos2[1]],      # p2 offset from origin
        ], dtype=np.float64)
        init_velocities = np.array([
            [0.0, 0.0],       # p1 held fixed
            velocity2,
        ], dtype=np.float64)

        # p1 controls field amplitude (source strength); p2 controls inertia.
        # Note: FieldSampler._paint_sources uses self.source_coupling as the
        # per-particle values array (not particle_source), so we pass our
        # per-particle source strengths through source_coupling.
        sim = FieldSampler(
            particle_inertia=np.array([1, p2]),     # p1 inertia ≫ 0 → effectively fixed
            particle_source=np.array([p1, 1.0]),       # stored but currently unused by step()
            particle_force=np.array([0.0, 1.0]),       # p1 feels no force (fixed)
            initial_positions=init_positions,
            initial_velocities=init_velocities,
            n_particles=2,
            spatial_dimensions=2,
            temporal_order=self.temporal_order,
            grid_size=self.grid_size,
            domain_size=self.domain_size,
            operators=self.operators,
            dt=self.dt,
            source_coupling=np.array([p1, 1.0]),       # per-particle, drives _paint_sources
            force_coupling=1.0,
            periodic_boundaries=True,
        )

        pos1_traj, pos2_traj = [], []
        vel1_traj, vel2_traj = [], []
        recorded = set()

        n_steps = int(round(duration / self.dt))
        for i in range(n_steps + 1):
            t = round(i * self.dt, 10)
            # Record at (or just past) each requested measurement time
            for mt in measurement_times:
                if mt not in recorded and t >= mt:
                    p1_pos = (sim.positions[0] - centre).tolist()
                    p2_pos = (sim.positions[1] - centre).tolist()
                    pos1_traj.append(p1_pos)
                    pos2_traj.append(p2_pos)
                    vel1_traj.append(sim.velocities[0].tolist())
                    vel2_traj.append(sim.velocities[1].tolist())
                    recorded.add(mt)
            if len(recorded) == len(measurement_times):
                break
            if i < n_steps:
                sim.step()

        return {
            "measurement_times": measurement_times,
            "pos1": pos1_traj,
            "pos2": pos2_traj,
            "velocity1": vel1_traj,
            "velocity2": vel2_traj,
        }


class CircleExecutor:
    """
    Runs 11-particle circle world experiments for the discovery agent.

    Layout: particle 0 at center, particles 1-10 equally spaced on a ring.
    The hidden physics is a fractional Laplacian with alpha=0.75.

    Experiment format:
        {
            "ring_radius": float,                   # ring radius (default 5.0)
            "initial_tangential_velocity": float,   # CCW tangential speed for ring (default 0.0)
            "measurement_times": [float, ...]
        }

    Returns:
        {
            "measurement_times": [...],
            "positions":  [[[x,y], ...], ...],   # shape (T, 11, 2), relative to domain center
            "velocities": [[[vx,vy], ...], ...]  # shape (T, 11, 2)
        }
    """

    N_RING = 10
    N_TOTAL = 11
    ALPHA = 0.75

    def __init__(
        self,
        operators=None,
        temporal_order=0,
        grid_size=(128, 128),
        domain_size=50.0,
        dt=0.005,
    ):
        self.operators = operators or [
            {"type": "fractional_laplacian", "params": {"strength": 1.0, "alpha": self.ALPHA}}
        ]
        self.temporal_order = temporal_order
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.dt = dt

    def run(self, experiments: list[dict]) -> list[dict]:
        return [self._run_one(exp) for exp in experiments]

    def run_json(self, json_str: str) -> str:
        experiments = json.loads(json_str)
        results = self.run(experiments)
        return json.dumps(results, indent=2)

    def _run_one(self, exp: dict) -> dict:
        ring_radius = float(exp.get("ring_radius", 5.0))
        v_tang = float(exp.get("initial_tangential_velocity", 0.0))
        measurement_times = sorted(exp["measurement_times"])
        duration = float(exp.get("duration", max(measurement_times)))
        duration = max(duration, 5.0)

        centre = self.domain_size / 2.0
        angles = np.linspace(0, 2 * np.pi, self.N_RING, endpoint=False)

        ring_pos = np.column_stack([
            centre + ring_radius * np.cos(angles),
            centre + ring_radius * np.sin(angles),
        ])
        positions = np.vstack([[[centre, centre]], ring_pos])

        # Tangential velocities: CCW perpendicular to radial direction
        ring_vel = np.column_stack([
            -v_tang * np.sin(angles),
             v_tang * np.cos(angles),
        ])
        velocities = np.vstack([[[0.0, 0.0]], ring_vel])

        masses = np.ones(self.N_TOTAL)
        sim = FieldSampler(
            particle_inertia=masses,
            particle_source=masses,
            particle_force=masses,
            initial_positions=positions,
            initial_velocities=velocities,
            n_particles=self.N_TOTAL,
            spatial_dimensions=2,
            temporal_order=self.temporal_order,
            grid_size=self.grid_size,
            domain_size=self.domain_size,
            operators=self.operators,
            dt=self.dt,
            source_coupling=masses,
            force_coupling=1.0,
            periodic_boundaries=False,
        )

        pos_traj, vel_traj = [], []
        recorded = set()

        n_steps = int(round(duration / self.dt))
        for i in range(n_steps + 1):
            t = round(i * self.dt, 10)
            for mt in measurement_times:
                if mt not in recorded and t >= mt:
                    pos_traj.append((sim.positions - centre).tolist())
                    vel_traj.append(sim.velocities.tolist())
                    recorded.add(mt)
            if len(recorded) == len(measurement_times):
                break
            if i < n_steps:
                sim.step()

        return {
            "measurement_times": measurement_times,
            "positions": pos_traj,   # (T, 11, 2) relative to domain center
            "velocities": vel_traj,  # (T, 11, 2)
        }


class SpeciesExecutor:
    """
    Runs 6-particle species world experiments for the discovery agent.

    Hidden structure: two species with different source couplings.
      Species A (particles 0, 1, 2): source_coupling = 1.0
      Species B (particles 3, 4, 5): source_coupling = 3.0

    All particles have equal mass (inertia = 1) and equal force coupling.
    The field is a standard Laplacian (n=0), so the *only* hidden variable
    is the per-particle source strength.

    Experiment format:
        {
            "positions":  [[x, y], ...],        # 6 initial positions (relative to center)
            "velocities": [[vx, vy], ...],      # 6 initial velocities
            "measurement_times": [float, ...]
        }

    Returns:
        {
            "measurement_times": [...],
            "positions":  [[[x,y], ...], ...],   # shape (T, 6, 2), relative to domain center
            "velocities": [[[vx,vy], ...], ...]  # shape (T, 6, 2)
        }
    """

    N_PARTICLES = 6
    SPECIES_A = [0, 1, 2]
    SPECIES_B = [3, 4, 5]
    SOURCE_A = 1.0
    SOURCE_B = 3.0

    def __init__(
        self,
        operators=None,
        temporal_order=0,
        grid_size=(64, 64),
        domain_size=20.0,
        dt=0.005,
    ):
        self.operators = operators or [
            {"type": "laplacian", "params": {"strength": 1.0}}
        ]
        self.temporal_order = temporal_order
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.dt = dt

    def run(self, experiments: list[dict]) -> list[dict]:
        return [self._run_one(exp) for exp in experiments]

    def run_json(self, json_str: str) -> str:
        experiments = json.loads(json_str)
        results = self.run(experiments)
        return json.dumps(results, indent=2)

    def _run_one(self, exp: dict) -> dict:
        positions_rel = np.array(exp["positions"], dtype=np.float64)
        velocities = np.array(exp["velocities"], dtype=np.float64)
        measurement_times = sorted(exp["measurement_times"])
        duration = float(exp.get("duration", max(measurement_times)))
        duration = max(duration, 5.0)

        assert positions_rel.shape == (self.N_PARTICLES, 2), \
            f"Expected {self.N_PARTICLES} positions, got {positions_rel.shape[0]}"
        assert velocities.shape == (self.N_PARTICLES, 2), \
            f"Expected {self.N_PARTICLES} velocities, got {velocities.shape[0]}"

        centre = self.domain_size / 2.0
        positions = positions_rel + centre  # shift to domain coords

        masses = np.ones(self.N_PARTICLES)
        source_coupling = np.ones(self.N_PARTICLES)
        source_coupling[self.SPECIES_B] = self.SOURCE_B

        sim = FieldSampler(
            particle_inertia=masses,
            particle_source=masses,
            particle_force=masses,
            initial_positions=positions,
            initial_velocities=velocities,
            n_particles=self.N_PARTICLES,
            spatial_dimensions=2,
            temporal_order=self.temporal_order,
            grid_size=self.grid_size,
            domain_size=self.domain_size,
            operators=self.operators,
            dt=self.dt,
            source_coupling=source_coupling,
            force_coupling=1.0,
            periodic_boundaries=True,
        )

        pos_traj, vel_traj = [], []
        recorded = set()

        n_steps = int(round(duration / self.dt))
        for i in range(n_steps + 1):
            t = round(i * self.dt, 10)
            for mt in measurement_times:
                if mt not in recorded and t >= mt:
                    pos_traj.append((sim.positions - centre).tolist())
                    vel_traj.append(sim.velocities.tolist())
                    recorded.add(mt)
            if len(recorded) == len(measurement_times):
                break
            if i < n_steps:
                sim.step()

        return {
            "measurement_times": measurement_times,
            "positions": pos_traj,    # (T, 6, 2) relative to domain center
            "velocities": vel_traj,   # (T, 6, 2)
        }

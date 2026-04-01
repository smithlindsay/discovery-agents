"""
Evaluator: compares the agent's discovered_law against ground-truth trajectories.
For now, just using raw MSE of the predicted vs PhysicsSchool particle paths
"""

import numpy as np
from typing import Callable

from scienceagent.executor import SimulationExecutor, SpeciesExecutor


# Default held-out test cases: (p1, p2, pos2, velocity2, measurement_times)
_DEFAULT_TEST_CASES = [
    {"p1": 1.0,  "p2": 1.0,  "pos2": [3.0, 0.0],  "velocity2": [0.0, 0.5],  "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]},
    {"p1": 2.0,  "p2": 1.0,  "pos2": [5.0, 0.0],  "velocity2": [0.0, 0.0],  "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]},
    {"p1": 1.0,  "p2": 2.0,  "pos2": [-4.0, 2.0], "velocity2": [0.3, -0.3], "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]},
]


class Evaluator:
    """
    Evaluates a discovered law function against the simulator.

    Args:
        executor: The same SimulationExecutor used during discovery (same world).
        test_cases: List of experiment dicts. Defaults to _DEFAULT_TEST_CASES.
    """

    def __init__(
        self,
        executor: SimulationExecutor,
        test_cases: list[dict] = None,
    ):
        self.executor = executor
        self.test_cases = test_cases or _DEFAULT_TEST_CASES

    def evaluate(self, law_source: str, verbose: bool = True) -> dict:
        """
        Execute the agent's law and compare against ground truth.

        Args:
            law_source: Python source string containing the `discovered_law` function.
            verbose: If True, print per-case results.

        Returns:
            dict with keys:
                mean_pos_error  — mean Euclidean position error across all (case, time) pairs
                max_pos_error   — max Euclidean position error
                per_case        — list of per-case mean position errors
                passed          — bool, True if mean_pos_error < 0.1 (10% of typical scale)
        """
        discovered_law = _compile_law(law_source)

        ground_truths = self.executor.run(self.test_cases)

        per_case_errors = []
        all_errors = []
        trajectories = []  # collected for plotting

        for i, (case, gt) in enumerate(zip(self.test_cases, ground_truths)):
            gt_pos1 = np.asarray(gt["pos1"])
            gt_pos2 = np.asarray(gt["pos2"])
            try:
                # Call discovered_law at each measurement time to build a full trajectory
                pred_traj = []
                case_errors = []
                vel = list(case["velocity2"])
                for j, t in enumerate(case["measurement_times"]):
                    p2_out, v2_out = discovered_law(
                        pos1=[0.0, 0.0],
                        pos2=case["pos2"],
                        p1=case["p1"],
                        p2=case["p2"],
                        velocity2=case["velocity2"],
                        duration=t,
                    )
                    p2_out = np.asarray(p2_out)
                    # If the law returns a trajectory array, take the last point
                    if p2_out.ndim == 2:
                        p2_out = p2_out[-1]
                    pred_traj.append(p2_out.tolist())
                    err = float(np.linalg.norm(p2_out - np.asarray(gt_pos2[j])))
                    case_errors.append(err)

                mean_err = float(np.mean(case_errors))
                per_case_errors.append(mean_err)
                all_errors.extend(case_errors)
                trajectories.append({
                    "case": i + 1,
                    "times": case["measurement_times"],
                    "p1": case["p1"], "p2": case["p2"],
                    "gt1": gt_pos1.tolist(),
                    "gt": gt_pos2.tolist(),
                    "pred": pred_traj,
                    "error": mean_err,
                })

                if verbose:
                    print(f"  Case {i+1}: mean_pos_error = {mean_err:.4f}")

            except Exception as e:
                if verbose:
                    print(f"  Case {i+1}: ERROR — {e}")
                per_case_errors.append(float("inf"))
                all_errors.append(float("inf"))
                trajectories.append({
                    "case": i + 1,
                    "times": case["measurement_times"],
                    "p1": case["p1"], "p2": case["p2"],
                    "gt1": gt_pos1.tolist(),
                    "gt": gt_pos2.tolist(),
                    "pred": None,
                    "error": float("inf"),
                })

        mean_total = float(np.mean(all_errors)) if all_errors else float("inf")
        max_total = float(np.max(all_errors)) if all_errors else float("inf")
        passed = mean_total < 0.1

        if verbose:
            print(f"\n  Mean position error: {mean_total:.4f}")
            print(f"  Max  position error: {max_total:.4f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")

        return {
            "mean_pos_error": mean_total,
            "max_pos_error": max_total,
            "per_case": per_case_errors,
            "passed": passed,
            "trajectories": trajectories,
        }


# seperate circle world for now as particle amounts are hard-coded into the evals
_CIRCLE_TEST_CASES = [
    # Ring with tangential velocity — tests orbital / spiral dynamics
    {"ring_radius": 5.0, "initial_tangential_velocity": 0.3,
     "measurement_times": [2.0, 4.0, 6.0, 8.0, 10.0]},
]


class CircleEvaluator:
    """
    Evaluates a discovered law for the 11-particle circle world.

    The discovered_law signature is:
        discovered_law(positions, velocities, duration) -> positions_final

    where:
        positions  — list of 11 [x, y] coords relative to center at t=0
        velocities — list of 11 [vx, vy] at t=0
        duration   — float, time to simulate
        return     — list/array of 11 [x, y] positions at t=duration

    The evaluator calls discovered_law once per measurement time (duration=t),
    computes Euclidean error per particle, and averages across all particles,
    times, and test cases.
    """

    def __init__(self, executor, test_cases: list[dict] = None):
        self.executor = executor
        self.test_cases = test_cases or _CIRCLE_TEST_CASES

    def evaluate(self, law_source: str, verbose: bool = True) -> dict:
        discovered_law = _compile_law(law_source)
        ground_truths = self.executor.run(self.test_cases)

        per_case_errors = []
        all_errors = []
        trajectories = []

        for i, (case, gt) in enumerate(zip(self.test_cases, ground_truths)):
            gt_positions = np.asarray(gt["positions"])   # (T, 11, 2)
            gt_velocities = np.asarray(gt["velocities"]) # (T, 11, 2)
            init_pos = gt_positions[0] if len(gt_positions) > 0 else None

            # Initial conditions: positions and velocities at t=0
            # Run the simulator one step from t=0 to get t=0 state
            zero_result = self.executor.run([{**case, "measurement_times": [case["measurement_times"][0]]}])
            # Actually just reconstruct from ring_radius and v_tang
            ring_radius = float(case.get("ring_radius", 5.0))
            v_tang = float(case.get("initial_tangential_velocity", 0.0))
            angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
            ring_pos = np.column_stack([ring_radius * np.cos(angles), ring_radius * np.sin(angles)])
            init_positions = np.vstack([[[0.0, 0.0]], ring_pos]).tolist()
            ring_vel = np.column_stack([-v_tang * np.sin(angles), v_tang * np.cos(angles)])
            init_velocities = np.vstack([[[0.0, 0.0]], ring_vel]).tolist()

            try:
                pred_traj = []
                case_errors = []
                for j, t in enumerate(case["measurement_times"]):
                    pos_out = discovered_law(
                        positions=init_positions,
                        velocities=init_velocities,
                        duration=t,
                    )
                    pos_out = np.asarray(pos_out)   # (11, 2)
                    pred_traj.append(pos_out.tolist())

                    # Per-particle Euclidean error
                    errs = np.linalg.norm(pos_out - gt_positions[j], axis=-1)  # (11,)
                    case_errors.append(float(np.mean(errs)))
                    all_errors.extend(errs.tolist())

                mean_err = float(np.mean(case_errors))
                per_case_errors.append(mean_err)
                trajectories.append({
                    "case": i + 1,
                    "ring_radius": case["ring_radius"],
                    "v_tang": case["initial_tangential_velocity"],
                    "times": case["measurement_times"],
                    "gt": gt_positions.tolist(),
                    "pred": pred_traj,
                    "error": mean_err,
                })
                if verbose:
                    print(f"  Case {i+1} (r={case['ring_radius']}, v_t={case['initial_tangential_velocity']}): "
                          f"mean_pos_error = {mean_err:.4f}")

            except Exception as e:
                if verbose:
                    print(f"  Case {i+1}: ERROR — {e}")
                per_case_errors.append(float("inf"))
                all_errors.append(float("inf"))
                trajectories.append({
                    "case": i + 1,
                    "ring_radius": case["ring_radius"],
                    "v_tang": case["initial_tangential_velocity"],
                    "times": case["measurement_times"],
                    "gt": gt_positions.tolist() if gt_positions is not None else [],
                    "pred": None,
                    "error": float("inf"),
                })

        mean_total = float(np.mean(all_errors)) if all_errors else float("inf")
        max_total = float(np.max(all_errors)) if all_errors else float("inf")
        passed = mean_total < 0.5   # looser threshold: 11-particle problem is harder

        if verbose:
            print(f"\n  Mean position error (all particles): {mean_total:.4f}")
            print(f"  Max  position error:                 {max_total:.4f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")

        return {
            "mean_pos_error": mean_total,
            "max_pos_error": max_total,
            "per_case": per_case_errors,
            "passed": passed,
            "trajectories": trajectories,
        }


# this is not working at all right now
_SPECIES_TEST_CASES = [
    # Asymmetric layout: species differences should cause divergent trajectories
    {
        "positions": [[0, 0], [4, 0], [-4, 0], [0, 4], [0, -4], [3, 3]],
        "velocities": [[0, 0], [0, 0.3], [0, -0.3], [0.3, 0], [-0.3, 0], [0, 0]],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0],
    },
    # Closer particles — stronger forces reveal species difference faster
    {
        "positions": [[0, 0], [2, 0], [-2, 0], [0, 2], [0, -2], [2, 2]],
        "velocities": [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        "measurement_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    },
]


class SpeciesEvaluator:
    """
    Evaluates a discovered law for the 6-particle species world.

    The discovered_law signature is:
        discovered_law(positions, velocities, duration) -> positions_final

    where:
        positions  -- list of 6 [x, y] coords relative to center at t=0
        velocities -- list of 6 [vx, vy] at t=0
        duration   -- float, time to simulate
        return     -- list/array of 6 [x, y] positions at t=duration
    """

    def __init__(self, executor, test_cases: list[dict] = None):
        self.executor = executor
        self.test_cases = test_cases or _SPECIES_TEST_CASES

    def evaluate(self, law_source: str, verbose: bool = True) -> dict:
        discovered_law = _compile_law(law_source)
        ground_truths = self.executor.run(self.test_cases)

        per_case_errors = []
        all_errors = []
        trajectories = []

        for i, (case, gt) in enumerate(zip(self.test_cases, ground_truths)):
            gt_positions = np.asarray(gt["positions"])    # (T, 6, 2)
            init_positions = case["positions"]
            init_velocities = case["velocities"]

            try:
                pred_traj = []
                case_errors = []
                for j, t in enumerate(case["measurement_times"]):
                    pos_out = discovered_law(
                        positions=init_positions,
                        velocities=init_velocities,
                        duration=t,
                    )
                    pos_out = np.asarray(pos_out)   # (6, 2)
                    pred_traj.append(pos_out.tolist())

                    errs = np.linalg.norm(pos_out - gt_positions[j], axis=-1)  # (6,)
                    case_errors.append(float(np.mean(errs)))
                    all_errors.extend(errs.tolist())

                mean_err = float(np.mean(case_errors))
                per_case_errors.append(mean_err)
                trajectories.append({
                    "case": i + 1,
                    "times": case["measurement_times"],
                    "gt": gt_positions.tolist(),
                    "pred": pred_traj,
                    "error": mean_err,
                })
                if verbose:
                    print(f"  Case {i+1}: mean_pos_error = {mean_err:.4f}")

            except Exception as e:
                if verbose:
                    print(f"  Case {i+1}: ERROR -- {e}")
                per_case_errors.append(float("inf"))
                all_errors.append(float("inf"))
                trajectories.append({
                    "case": i + 1,
                    "times": case["measurement_times"],
                    "gt": gt_positions.tolist(),
                    "pred": None,
                    "error": float("inf"),
                })

        mean_total = float(np.mean(all_errors)) if all_errors else float("inf")
        max_total = float(np.max(all_errors)) if all_errors else float("inf")
        passed = mean_total < 0.3

        if verbose:
            print(f"\n  Mean position error (all particles): {mean_total:.4f}")
            print(f"  Max  position error:                 {max_total:.4f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")

        return {
            "mean_pos_error": mean_total,
            "max_pos_error": max_total,
            "per_case": per_case_errors,
            "passed": passed,
            "trajectories": trajectories,
        }

# -----------------
# utility functions
def clean_law_source(source: str) -> str:
    """Strip markdown fences and prose before the first code line."""
    import re as _re
    source = _re.sub(r"^```[a-zA-Z]*\n?", "", source.strip(), flags=_re.MULTILINE)
    source = source.replace("```", "")
    lines = source.splitlines()
    code_start = next(
        (i for i, l in enumerate(lines) if l.startswith("def ") or l.startswith("import ") or l.startswith("from ")),
        0,
    )
    return "\n".join(lines[code_start:])


def _compile_law(source: str) -> Callable:
    """Compile and return the discovered_law function from a source string."""
    source = clean_law_source(source)
    namespace = {}
    exec(compile(source, "<discovered_law>", "exec"), namespace)
    if "discovered_law" not in namespace:
        raise ValueError("Source does not define a function named `discovered_law`")
    return namespace["discovered_law"]

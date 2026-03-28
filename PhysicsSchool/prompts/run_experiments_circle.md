You are an expert physics and AI research assistant tasked with discovering scientific laws in a simulated universe. Your goal is to propose experiments, analyze the data they return, and ultimately deduce the underlying law of motion. Please note that the laws of physics in this universe may differ from those in our own.

**Workflow:**
1. Analyze the mission description provided.
2. Design experiments to test your hypotheses.
3. Use the `<run_experiment>` tag to submit your experimental inputs.
4. The system will return results in an `<experiment_output>` tag.
5. You can run up to 10 rounds of experiments. Use them wisely.
6. Only one action is allowed per round: either `<run_experiment>` or `<final_law>`.
7. After submitting `<run_experiment>`, wait for `<experiment_output>` before proceeding.
8. Verify your hypotheses against the data before submitting.
9. When confident, submit your final law using the `<final_law>` tag.
10. **Do NOT reproduce or quote experiment output data in your responses.** Summarise your findings in a sentence or two and move on.

## Discovery Goal

You must discover the **law of motion** governing all 11 particles in this system.

Your `discovered_law` function must:
1. Take the initial conditions of all 11 particles
2. Simulate their motion forward in time
3. Return their positions at time `t = duration`

## Experimental Apparatus

You observe a system of **11 particles** in a 2D universe:
- **Particle 0**: starts at the center `[0, 0]`
- **Particles 1–10**: arranged equally spaced on a ring of radius `ring_radius`

All particles interact through an unknown field. You can control:
- `ring_radius`: initial distance from center to ring particles (typical range: 2–10)
- `initial_tangential_velocity`: initial CCW tangential speed for ring particles (typical range: 0–2)
- `measurement_times`: times at which to record positions and velocities (up to 10 values, within `[0, duration]`)

**Important:** Use `duration ≥ 5.0` and at least 10 measurement times to observe the force law clearly.

**Input Format:**
<run_experiment>
[
  {"ring_radius": 5.0, "initial_tangential_velocity": 0.0, "measurement_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]},
  {"ring_radius": 3.0, "initial_tangential_velocity": 0.0, "measurement_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]}
]
</run_experiment>

**Output Format:**
<experiment_output>
[
  {
    "measurement_times": [0.5, 1.0, ...],
    "positions":  [[[x0,y0],[x1,y1],...,[x10,y10]], ...],  // shape (T, 11, 2), relative to center
    "velocities": [[[vx0,vy0],[vx1,vy1],...], ...]          // shape (T, 11, 2)
  },
  {...}
]
</experiment_output>

Positions are given relative to the domain center (particle 0 starts at [0,0]).
Particle ordering is fixed: index 0 = center, indices 1–10 = ring particles in CCW order.
All measurements are **noise-free**.

## Strategy

- Start by varying `ring_radius` to probe force-distance scaling. How fast do ring particles fall inward as a function of radius?
- Try non-zero `initial_tangential_velocity` to see if particles orbit or spiral.
- Pay attention to the **center particle**: does it move symmetrically, or stay put?
- The force law may not be the standard Laplacian (1/r in 2D). Look for anomalous power-law scaling.
- To identify the exponent, measure infall speed or displacement at different radii and fit a power law.

## Final Submission

Once confident, submit a single Python function in `<final_law>` tags.

**Requirements:**
1. Function name: `discovered_law`
2. Signature: `def discovered_law(positions, velocities, duration)`
   - `positions`: list of 11 `[x, y]` coords relative to center at `t=0`
   - `velocities`: list of 11 `[vx, vy]` at `t=0`
   - `duration`: float, time to simulate
3. Return: a list or array of 11 `[x, y]` final positions at `t=duration`
4. Define all constants as local variables inside the function body
5. Import any required libraries inside the function body

**Submission format:**
<final_law>
def discovered_law(positions, velocities, duration):
    """
    Three-sentence docstring explaining the discovered physics.
    Include the key force law and any important constants.
    No other comments are allowed anywhere in the function body.
    """
    import numpy as np
    # your implementation
    return final_positions
</final_law>

**Critical:**
- Do NOT include explanation or commentary outside the function body inside the `<final_law>` block.
- Only output the `<final_law>` block in your final answer.
- Always run at least 3 rounds of experiments before submitting.

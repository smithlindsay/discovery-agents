#!/usr/bin/env python
"""
CLI runner for a single physics-discovery episode.
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Run a physics discovery episode.")
    parser.add_argument("--model", default="claude-opus-4-6", help="LLM model string")
    parser.add_argument("--world", default="gravity", help="World name (see worlds.py)")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--quiet", action="store_true", help="Suppress per-round output")
    parser.add_argument("--show-experiment-output", action="store_true", help="Print simulator experiment output each round")
    parser.add_argument("--output", default=None, help="Write results JSON to this file")
    parser.add_argument("--plot", default=None, help="Save trajectory comparison plot to this path (e.g. plot.png)")
    parser.add_argument("--store_output", default=None,
                        help="Base path for full run logs: writes <path>.json and <path>.txt")
    args = parser.parse_args()

    from scienceagent.worlds import get_world
    from scienceagent.agent import DiscoveryAgent
    from scienceagent.evaluator import Evaluator, CircleEvaluator, SpeciesEvaluator, clean_law_source

    print(f"World : {args.world}")
    print(f"Model : {args.model}")
    print()

    world = get_world(args.world)
    executor       = world["executor"]
    mission        = world["mission"]
    true_law       = world["true_law"]
    true_law_title = world["true_law_title"]

    agent = DiscoveryAgent(
        model=args.model,
        executor=executor,
        mission=mission,
        max_tokens=args.max_tokens,
        verbose=not args.quiet,
        show_experiment_output=args.show_experiment_output,
        system_prompt_path=world["system_prompt"],
        law_stub=world["law_stub"],
        experiment_format=world["experiment_format"],
    )

    law_source = agent.run()

    if law_source is None:
        print("\n[No final law submitted]")
        return

    print("\n" + "="*60)
    print("Discovered law:")
    print("="*60)
    print(law_source)

    print("\n" + "="*60)
    print("Evaluation:")
    print("="*60)
    if args.world == "circle":
        evaluator = CircleEvaluator(executor)
    elif args.world == "species":
        evaluator = SpeciesEvaluator(executor)
    else:
        evaluator = Evaluator(executor)
    results = evaluator.evaluate(law_source, verbose=True)

    if args.plot:
        base = args.plot
        plot_path = base + "_plot.png"
        law_path  = base + "_law.png"
        if args.world == "circle":
            _plot_circle_trajectories(results["trajectories"], args.world, args.model, plot_path)
            _plot_circle_law(clean_law_source(law_source), args.world, args.model, law_path)
        elif args.world == "species":
            _plot_species_trajectories(results["trajectories"], args.world, args.model, plot_path)
            _plot_law(clean_law_source(law_source), args.world, args.model, law_path)
        else:
            _plot_trajectories(results["trajectories"], args.world, args.model, plot_path)
            _plot_law(clean_law_source(law_source), args.world, args.model, law_path)

    if args.output:
        out = {"world": args.world, "model": args.model, "law": law_source, "evaluation": results}
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to {args.output}")

    if args.store_output:
        import datetime
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        _write_run_json(args.store_output + ".json", args.world, args.model, timestamp,
                        agent, law_source, results)
        _write_run_txt(args.store_output + ".txt",  args.world, args.model, timestamp,
                       agent, law_source, results)
        print(f"\nFull run log written to {args.store_output}.json / .txt")


def _write_run_json(path, world, model, timestamp, agent, law_source, evaluation):
    """Write the full structured run log as JSON."""
    data = {
        "world": world,
        "model": model,
        "timestamp": timestamp,
        "mission": agent.mission,
        "rounds": agent.conversation_log,
        "final_law": law_source,
        "evaluation": evaluation,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_run_txt(path, world, model, timestamp, agent, law_source, evaluation):
    """Write the full run log as a human-readable plain-text file."""
    lines = []
    sep = "=" * 60

    lines += [
        sep,
        "PHYSICS DISCOVERY RUN",
        sep,
        f"World     : {world}",
        f"Model     : {model}",
        f"Timestamp : {timestamp}",
        f"Mission   : {agent.mission or '(none)'}",
        "",
    ]

    for entry in agent.conversation_log:
        lines += [sep, f"ROUND {entry['round']}", sep]

        if entry.get("system_message"):
            lines += ["[System]", entry["system_message"], ""]

        if entry.get("llm_reply"):
            lines += ["[LLM Reply]", entry["llm_reply"], ""]

        action = entry.get("action")
        if action == "experiment":
            if entry.get("experiment_input") is not None:
                lines += ["[Experiment Input]",
                          json.dumps(entry["experiment_input"], indent=2), ""]
            if entry.get("experiment_output") is not None:
                lines += ["[Experiment Output]",
                          json.dumps(entry["experiment_output"], indent=2), ""]
            if entry.get("experiment_error"):
                lines += ["[Experiment Error]", entry["experiment_error"], ""]
        elif action == "final_law":
            lines += ["[Final Law Submitted]", entry.get("final_law", ""), ""]
        elif action in ("warning", "no_tag"):
            lines += [f"[{action.upper()}]", entry.get("system_message", ""), ""]

    lines += [sep, "DISCOVERED LAW", sep, law_source or "(none)", ""]

    lines += [sep, "EVALUATION", sep]
    if evaluation:
        for case in evaluation.get("per_case", []):
            lines.append(f"  Case error: {case:.4f}")
        lines += [
            f"  Mean position error : {evaluation.get('mean_pos_error', float('inf')):.4f}",
            f"  Max  position error : {evaluation.get('max_pos_error',  float('inf')):.4f}",
            f"  Result              : {'PASS' if evaluation.get('passed') else 'FAIL'}",
        ]
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def _plot_trajectories(trajectories, world, model, path):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from matplotlib.gridspec import GridSpec

    mpl.rcParams["font.family"] = "serif"

    n = len(trajectories)
    fig = plt.figure(figsize=(4.5 * n, 6))
    gs = GridSpec(1, n, figure=fig, wspace=0.45,
                  left=0.05, right=0.97, top=0.88, bottom=0.12)

    for j, (ax, traj) in enumerate(zip([fig.add_subplot(gs[0, j]) for j in range(n)], trajectories)):
        gt = np.asarray(traj["gt"])
        ax.plot(gt[:, 0], gt[:, 1], "o-", color="black", label="Ground truth", zorder=3)
        if traj["pred"] is not None:
            pred = np.asarray(traj["pred"])
            ax.plot(pred[:, 0], pred[:, 1], "s--", color="darkorange", label="Predicted", zorder=4)
        ax.set_title(f"Case {traj['case']}  (p1={traj['p1']}, p2={traj['p2']})\nerr={traj['error']:.4f}", fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.minorticks_on()

    model_short = os.path.basename(model)
    fig.suptitle(f"World: {world}  |  Model: {model_short}", fontsize=11, y=0.98)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")


def _plot_circle_trajectories(trajectories, world, model, path):
    """Plot ring particle trajectories (GT then pred on top) for the circle world."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np

    mpl.rcParams["font.family"] = "serif"

    traj = trajectories[0]
    gt   = np.asarray(traj["gt"])
    pred = np.asarray(traj["pred"]) if traj["pred"] is not None else None

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.1)

    for i in range(1, gt.shape[1]):
        ax.plot(gt[:, i, 0], gt[:, i, 1], "-", color="black", lw=1.2,
                alpha=0.8, label="Ground truth" if i == 1 else None, zorder=3)
    ax.scatter(gt[-1, 1:, 0], gt[-1, 1:, 1],
               color="black", s=40, zorder=5, marker="o", edgecolors="none")

    if pred is not None:
        for i in range(1, pred.shape[1]):
            ax.plot(pred[:, i, 0], pred[:, i, 1], "--", color="tomato", lw=1.2,
                    alpha=0.9, label="Predicted" if i == 1 else None, zorder=4)
        ax.scatter(pred[-1, 1:, 0], pred[-1, 1:, 1],
                   color="#E65C00", s=60, zorder=6, marker="x", linewidths=1.5)

    r, v, err = traj.get("ring_radius", "?"), traj.get("v_tang", "?"), traj["error"]
    ax.set_title(f"Circle world  (r={r}, v_tang={v})\nmean err = {err:.4f}", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.minorticks_on()

    model_short = os.path.basename(model)
    fig.suptitle(f"World: {world}  |  Model: {model_short}", fontsize=11, y=0.98)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")


def _plot_species_trajectories(trajectories, world, model, path):
    """Plot 6-particle species world: species A (blue) and B (green) GT, single-color predictions."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np

    mpl.rcParams["font.family"] = "serif"

    n_cases = len(trajectories)
    fig, axes = plt.subplots(1, n_cases, figsize=(5.5 * n_cases, 5.5), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.08, wspace=0.35)

    color_a = "#2166ac"   # blue for species A
    color_b = "#1a9850"   # green for species B
    color_pred = "tomato"
    species_a = [0, 1, 2]
    species_b = [3, 4, 5]

    for idx, traj in enumerate(trajectories):
        ax = axes[0, idx]
        gt = np.asarray(traj["gt"])       # (T, 6, 2)
        pred = np.asarray(traj["pred"]) if traj["pred"] is not None else None

        # Ground truth trajectories, colored by species
        for i in species_a:
            ax.plot(gt[:, i, 0], gt[:, i, 1], "-", color=color_a, lw=1.2, alpha=0.8,
                    label="Species A (GT)" if i == species_a[0] else None, zorder=3)
        for i in species_b:
            ax.plot(gt[:, i, 0], gt[:, i, 1], "-", color=color_b, lw=1.2, alpha=0.8,
                    label="Species B (GT)" if i == species_b[0] else None, zorder=3)

        # Mark initial positions with particle index labels
        for i in range(6):
            c = color_a if i in species_a else color_b
            ax.scatter(gt[0, i, 0], gt[0, i, 1], color=c, s=50, zorder=7,
                       marker="o", edgecolors="white", linewidths=0.5)
            ax.annotate(str(i), (gt[0, i, 0], gt[0, i, 1]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color=c, fontweight="bold", zorder=8)

        # Mark final GT positions
        for i in range(6):
            c = color_a if i in species_a else color_b
            ax.scatter(gt[-1, i, 0], gt[-1, i, 1], color=c, s=40, zorder=6,
                       marker="D", edgecolors="white", linewidths=0.5)

        # Predicted trajectories (single color)
        if pred is not None:
            for i in range(pred.shape[1]):
                ax.plot(pred[:, i, 0], pred[:, i, 1], "--", color=color_pred, lw=1.0,
                        alpha=0.85, label="Predicted" if i == 0 else None, zorder=4)
            ax.scatter(pred[-1, :, 0], pred[-1, :, 1],
                       color="#E65C00", s=50, zorder=6, marker="x", linewidths=1.5)

        ax.set_title(f"Case {traj['case']}  |  mean err = {traj['error']:.4f}", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8, loc="best")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.minorticks_on()

    model_short = os.path.basename(model)
    fig.suptitle(f"World: {world}  |  Model: {model_short}", fontsize=11, y=0.98)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")


def _plot_circle_law(law_source, world, model, path):
    """Render the discovered law as a standalone figure."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "serif"

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axis("off")
    ax.text(
        0.02, 0.5, law_source or "(no law submitted)",
        transform=ax.transAxes,
        fontsize=8, verticalalignment="center", horizontalalignment="left",
        family="monospace", color="#1a3a6b",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#d8d8d8", edgecolor="#aaaaaa"),
    )
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")


def _plot_law(law_source, world, model, path):
    """Render the discovered law as a standalone figure (non-circle worlds)."""
    _plot_circle_law(law_source, world, model, path)


if __name__ == "__main__":
    main()

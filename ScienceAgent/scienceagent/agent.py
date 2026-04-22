"""
DiscoveryAgent: runs the experiment-design loop until <final_law> is submitted.

The agent follows the protocol defined in PhysicsSchool/prompts/run_experiments.md:
  - Submits experiments via <run_experiment>...</run_experiment> XML tags
  - Receives results in <experiment_output>...</experiment_output> tags
  - Submits its discovered law in <final_law>...</final_law> tags
  - Maximum MAX_ROUNDS rounds before the loop is terminated

Note, there is also a seperate supervisor/critic class to help keep the agent in line 
"""

import json
import re
from typing import Callable, Optional

from scienceagent import llm_client
from scienceagent.executor import SimulationExecutor

# somewhat arbitrary for now, more round budget could help, however the experiments
# must remain useful
MAX_ROUNDS = 10
MIN_ROUNDS = 2

# always give the same default instructions to the agent
_SYSTEM_PROMPT_PATH = "PhysicsSchool/prompts/run_experiments.md"

# rough format of a "law"
_DEFAULT_LAW_STUB = (
    "def discovered_law(pos1, pos2, p1, p2, velocity2, duration):\n"
    "    # your best implementation\n"
    "    return final_pos2, final_vel2\n"
)

# rough JSON style experiment action
_DEFAULT_EXPERIMENT_FORMAT = (
    "<run_experiment>[{\"p1\": 1.0, \"p2\": 1.0, \"pos2\": [3.0, 0.0], "
    "\"velocity2\": [0.0, 0.0], \"measurement_times\": [0.5, 1.0, 2.0]}]</run_experiment>"
)


def _load_system_prompt(prompt_path: str = None) -> str:
    import os
    prompt_path = prompt_path or _SYSTEM_PROMPT_PATH
    here = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    path = os.path.join(root, prompt_path)
    if not os.path.exists(path):
        # Fallback: assume CWD is repo root
        path = prompt_path
    with open(path) as f:
        return f.read()


class DiscoveryAgent:
    """
    Drives an LLM through the physics-discovery experiment loop.

    Args:
        model: Model string passed to llm_client.complete().
        executor: SimulationExecutor instance wrapping the target world.
        mission: Optional mission description appended to the system prompt.
            Describes what the agent should discover without revealing the answer.
        max_tokens: Max tokens per LLM call.
        verbose: If True, print each round's exchange to stdout.
        show_experiment_output: If True, also print the simulator's experiment output each round.
    """

    def __init__(
        self,
        model: str,
        executor,
        mission: Optional[str] = None,
        max_tokens: int = 4096,
        verbose: bool = True,
        show_experiment_output: bool = False,
        system_prompt_path: str = None,
        law_stub: str = None,
        experiment_format: str = None,
        critic=None,
        max_rounds: int = MAX_ROUNDS,
        min_rounds: int = MIN_ROUNDS,
        random_experiments: bool = False,
        random_generator: Optional[Callable[[], dict]] = None,
    ):
        self.model = model
        self.executor = executor
        self.mission = mission
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.show_experiment_output = show_experiment_output
        self.critic = critic
        self._system_prompt_path = system_prompt_path or _SYSTEM_PROMPT_PATH
        self._law_stub = law_stub or _DEFAULT_LAW_STUB
        self._experiment_format = experiment_format or _DEFAULT_EXPERIMENT_FORMAT
        self.max_rounds = max(1, int(max_rounds))
        self.min_rounds = max(1, min(int(min_rounds), self.max_rounds))
        self.random_experiments = bool(random_experiments)
        self.random_generator = random_generator
        if self.random_experiments and self.random_generator is None:
            raise ValueError(
                "random_experiments=True requires a random_generator callable."
            )
        self._system = self._build_system_prompt()
        # Populated during run(); each entry is a dict describing one round.
        self.conversation_log: list[dict] = []
        # Populated when the agent submits its final law alongside an <explanation> tag.
        self.discovered_explanation: Optional[str] = None

    def run(self) -> Optional[str]:
        """
        Run the discovery loop.

        Returns:
            The discovered law as a Python source string, or None if the agent
            did not submit a final law within MAX_ROUNDS.

        """
        self.conversation_log = []
        self.discovered_explanation = None
        messages = []
        if self.mission:
            messages.append({"role": "user", "content": self.mission})

        for round_num in range(1, self.max_rounds + 1):
            if self.verbose:
                print(f"\n{'='*52}")
                print(f"󰙨 󰧑  science agent experimenting at round {round_num}/{self.max_rounds} 󰧑  󰙨")
                print(f"{'='*52}")

            round_entry = {
                "round": round_num,
                "system_message": None,
                "llm_reply": None,
                "action": None,            # "experiment" | "final_law" | "warning" | "no_tag" | "random_experiment"
                "experiment_input": None,  # parsed JSON list, or None
                "experiment_output": None, # parsed JSON list, or None
                "experiment_error": None,
                "final_law": None,
                "explanation": None,
                "critic_feedback": None,
            }

            # In random-experiments mode, draw + execute the per-round experiment
            # BEFORE prompting the LLM, so the agent sees fresh data each turn.
            if self.random_experiments:
                self._inject_random_experiment(round_num, messages, round_entry)

            # On the second-to-last round, warn the agent it must submit next round
            if self.max_rounds >= 2 and round_num == self.max_rounds - 1:
                warn_msg = (
                    f"Warning: this is round {round_num} of {self.max_rounds}. "
                    "You have one round remaining. Your next response MUST be a <final_law> submission — "
                    "you will not be able to run further experiments after this."
                )
                messages.append({"role": "user", "content": warn_msg})
                round_entry["system_message"] = _join_sys(round_entry["system_message"], warn_msg)

            # On the final round, force a final_law submission
            if round_num == self.max_rounds:
                force_msg = (
                    "This is your final round. You MUST now submit your best guess "
                    "as a <final_law> regardless of confidence. Do not run more experiments.\n\n"
                    "<final_law>\n"
                    + self._law_stub
                    + "</final_law>"
                )
                messages.append({"role": "user", "content": force_msg})
                round_entry["system_message"] = _join_sys(round_entry["system_message"], force_msg)

            reply = llm_client.complete(
                model=self.model,
                messages=messages,
                system=self._system,
                max_tokens=self.max_tokens,
            )
            round_entry["llm_reply"] = reply

            if self.verbose:
                print(f"\n[Science Agent]\n{reply}")

            messages.append({"role": "assistant", "content": reply})

            # Check for final law submission
            final_law = _extract_tag(reply, "final_law")
            if final_law is not None:
                if round_num >= self.min_rounds:
                    if self.verbose:
                        print("\n[Agent submitted final law]")
                    explanation = _extract_tag(reply, "explanation")
                    # If the agent submitted a law without the required explanation,
                    # re-prompt once for the explanation only.
                    if explanation is None:
                        if self.verbose:
                            print("[Warning] Final law submitted without <explanation>. Re-prompting once.")
                        followup_msg = (
                            "Your <final_law> submission is accepted, but you did not include "
                            "the required <explanation> tag. Reply NOW with ONLY a single "
                            "<explanation>...</explanation> block containing a 2–3 sentence "
                            "plain-English description of the physical system you discovered. "
                            "Do not include any other text or tags."
                        )
                        messages.append({"role": "user", "content": followup_msg})
                        followup_reply = llm_client.complete(
                            model=self.model,
                            messages=messages,
                            system=self._system,
                            max_tokens=self.max_tokens,
                        )
                        messages.append({"role": "assistant", "content": followup_reply})
                        if self.verbose:
                            print(f"\n[Science Agent — explanation follow-up]\n{followup_reply}")
                        explanation = _extract_tag(followup_reply, "explanation")
                    self.discovered_explanation = explanation.strip() if explanation else None
                    round_entry["action"] = "final_law"
                    round_entry["final_law"] = final_law.strip()
                    round_entry["explanation"] = self.discovered_explanation
                    self.conversation_log.append(round_entry)
                    return final_law.strip()
                else:
                    if self.random_experiments:
                        warn = (
                            f"You have only seen {round_num} round(s) of random experiments. "
                            f"You must wait for at least {self.min_rounds} rounds before submitting a final law. "
                            "Another random experiment will be generated next round — please wait."
                        )
                    else:
                        warn = (
                            f"You have only run {round_num} round(s) of experiments. "
                            f"You must run at least {self.min_rounds} rounds before submitting a final law. "
                            "Please design and run at least one more experiment."
                        )
                    if self.verbose:
                        print(f"[Warning] Agent submitted final law too early (round {round_num}/{self.min_rounds} minimum). Requiring more experiments.")
                    round_entry["action"] = "warning"
                    round_entry["system_message"] = warn
                    self.conversation_log.append(round_entry)
                    messages.append({"role": "user", "content": warn})
                    continue

            # Random-experiments mode never parses <run_experiment>: the agent's
            # reply is treated as analysis/acknowledgement, the experiment for
            # the NEXT round was already drawn at the top of the loop iteration.
            if self.random_experiments:
                if round_entry["action"] is None:
                    round_entry["action"] = "random_experiment"
                self.conversation_log.append(round_entry)
                continue

            # Check for experiment request
            experiment_block = _extract_tag(reply, "run_experiment")
            if experiment_block is None:
                if self.verbose:
                    print("[Warning] No recognized tag in response. Prompting agent to continue.")
                no_tag_msg = (
                    "ERROR: No <run_experiment> or <final_law> tag found in your response. "
                    "You must respond with EXACTLY one of these XML tags — no code fences, "
                    "no markdown, just the raw tag.\n\n"
                    "Option 1 — run an experiment:\n"
                    + self._experiment_format + "\n\n"
                    + "Option 2 — submit your final law:\n"
                    "<final_law>\n"
                    + self._law_stub
                    + "</final_law>\n\n"
                    "Respond with the XML tag NOW. No explanation before or after."
                )
                round_entry["action"] = "no_tag"
                round_entry["system_message"] = no_tag_msg
                self.conversation_log.append(round_entry)
                messages.append({"role": "user", "content": no_tag_msg})
                continue

            # Run the experiments
            try:
                exp_input = json.loads(experiment_block)
                results = self.executor.run(exp_input)
                output_content = (
                    "<experiment_output>\n"
                    + _compact_json(results)
                    + "\n</experiment_output>"
                )
                round_entry["action"] = "experiment"
                round_entry["experiment_input"] = exp_input
                round_entry["experiment_output"] = results
            except Exception as e:
                output_content = (
                    f"<experiment_output>\nError running experiment: {e}\n</experiment_output>"
                )
                round_entry["action"] = "experiment"
                round_entry["experiment_error"] = str(e)

            if self.verbose and self.show_experiment_output:
                print(f"\n[Simulator]\n{output_content}")

            messages.append({"role": "user", "content": output_content})

            # Critic feedback injection (skip round 1)
            # this seems to help when the critic model is strong
            if self.critic and round_num >= 2 and round_entry["action"] == "experiment":
                critic_feedback = self.critic.review(
                    agent_system_prompt=self._system,
                    messages=messages,
                    round_num=round_num,
                )
                if self.verbose:
                    print(f"\n[Supervisor Agent]\n{critic_feedback}")
                critic_msg = (
                    f"Supervisor feedback:\n{critic_feedback}\n\n"
                    "Before proceeding with your next experiment or final law submission, "
                    "briefly acknowledge the supervisor's feedback above: state what you will "
                    "change or why you disagree. Then continue with your action."
                )
                messages.append({"role": "user", "content": critic_msg})
                round_entry["critic_feedback"] = critic_feedback

            self.conversation_log.append(round_entry)

        if self.verbose:
            print(f"\n[Agent did not submit a final law within {self.max_rounds} rounds]")
        return None

    def _inject_random_experiment(self, round_num: int, messages: list, round_entry: dict) -> None:
        """Draw one random experiment, run it, and append the output to messages.

        Mutates `messages` (adds a user turn containing the preamble + results)
        and `round_entry` (records experiment_input/output and a preamble as
        system_message). Called at the top of every round when
        self.random_experiments is True.
        """
        exp_input = self.random_generator()
        try:
            results = self.executor.run([exp_input])
            output_body = _compact_json(results)
            round_entry["experiment_input"] = exp_input
            round_entry["experiment_output"] = results
        except Exception as e:
            output_body = f"Error running experiment: {e}"
            round_entry["experiment_input"] = exp_input
            round_entry["experiment_error"] = str(e)

        round_entry["action"] = "random_experiment"

        preamble = (
            f"A random experiment ({round_num} of {self.max_rounds}) has been "
            "drawn and executed automatically. The inputs and resulting "
            "trajectory are below. Briefly acknowledge what this data tells "
            "you. Do NOT output a <run_experiment> tag — the next random "
            "experiment will be generated on the next round regardless of "
            "what you ask for."
        )
        user_block = (
            preamble
            + "\n\n<experiment_input>\n" + json.dumps(exp_input) + "\n</experiment_input>"
            + "\n<experiment_output>\n" + output_body + "\n</experiment_output>"
        )
        messages.append({"role": "user", "content": user_block})
        round_entry["system_message"] = _join_sys(round_entry.get("system_message"), preamble)

        if self.verbose and self.show_experiment_output:
            print(f"\n[Simulator — random experiment {round_num}]\n{user_block}")

    def _build_system_prompt(self) -> str:
        try:
            base = _load_system_prompt(self._system_prompt_path)
        except FileNotFoundError:
            base = (
                "You are a scientific discovery agent. Design experiments, "
                "analyze results, and discover the underlying law of physics."
            )
        return base.rstrip() + "\n\n" + self._run_policy_note()

    def _run_policy_note(self) -> str:
        """Run-specific footer injected after the world's system prompt.

        Overrides any fixed round budgets baked into the prompt files and
        pins down the minimum integration timestep the agent is allowed to
        use inside `discovered_law`.
        """
        return (
            "## RUN-SPECIFIC CONSTRAINTS (these override any conflicting numbers above)\n"
            f"- For this session you have EXACTLY {self.max_rounds} round(s) of "
            f"experiments. Plan accordingly: on round {self.max_rounds} you will be "
            "forced to submit your <final_law>.\n"
            f"- You must run at least {self.min_rounds} round(s) of experiments "
            "before submitting a <final_law>.\n"
            "- If your `discovered_law` integrates the trajectory with a for-loop "
            "over some timestep `dt`, the SMALLEST value of `dt` you are allowed "
            "to use is 0.01. Do not set dt below 0.01 under any circumstances.\n"
            "- ALWAYS set `dt > 0.005` in EVERY test simulation AND in your final "
            "law — no timestep anywhere in your code may be 0.005 or smaller.\n"
        )


def _join_sys(existing: Optional[str], new: str) -> str:
    """Concatenate system-message fragments on a single round_entry."""
    if not existing:
        return new
    return existing + "\n\n" + new


def _round_floats(obj, decimals=4):
    """Recursively round floats in nested lists/dicts."""
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, list):
        return [_round_floats(v, decimals) for v in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    return obj


def _compact_json(results) -> str:
    """Serialize experiment results with rounded floats and minimal whitespace."""
    return json.dumps(_round_floats(results), separators=(",", ":"))


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences that some models wrap XML tags in."""
    # Strip ```xml ... ```, ```python ... ```, or bare ``` ... ```
    text = re.sub(r"```(?:xml|python|json)?\s*\n?", "", text)
    text = re.sub(r"```\s*", "", text)
    return text


def _extract_tag(text: str, tag: str) -> Optional[str]:
    """Return the content between <tag>...</tag>, or None if not present."""
    if not text:
        return None
    # Try raw text first, then with code fences stripped
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    cleaned = _strip_code_fences(text)
    match = re.search(pattern, cleaned, re.DOTALL)
    return match.group(1) if match else None

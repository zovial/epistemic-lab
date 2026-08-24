from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .cfr import CFRSolver, exploitability
from .leduc import Action, LeducState
from .models import (
    BayesianOpponentModeler,
    Hypothesis,
    ScriptedOpponent,
    brute_force_hypothesis_posterior,
    default_hypotheses,
)
from .simulation import predictive_log_likelihood, sample_observations


def run_all(out: Path, cfr_iterations: int = 100) -> None:
    out.mkdir(parents=True, exist_ok=True)
    duhem_quine(out)
    merging(out)
    popper_vs_bayes(out)
    lakatos(out)
    dutch_book(out)
    cfr_baseline(out, cfr_iterations)


def duhem_quine(out: Path) -> None:
    model_a = ScriptedOpponent("same_on_path_a", aggression=0.42, looseness=0.55, raise_bias=0.2, off_path_probe_response=0.15)
    model_b = ScriptedOpponent("same_on_path_b", aggression=0.42, looseness=0.55, raise_bias=0.2, off_path_probe_response=0.9)
    on_path = sample_observations(
        model_a,
        200,
        predicate=lambda s: "r" not in s.history,
    )
    off_path_state = (
        LeducState.initial()
        ._with_private(0, 4)
        .apply_action(Action.CHECK)
        .apply_action(Action.BET)
        .apply_action(Action.RAISE)
    )
    on_path_delta = abs(
        predictive_log_likelihood(model_a, on_path) - predictive_log_likelihood(model_b, on_path)
    )
    off_path_probs_a = model_a.action_probs(off_path_state, 1)
    off_path_probs_b = model_b.action_probs(off_path_state, 1)
    rows = [
        ["metric", "value"],
        ["abs_on_path_loglik_delta_200_obs", on_path_delta],
        ["off_path_call_prob_model_a", off_path_probs_a.get(Action.CALL, 0.0)],
        ["off_path_call_prob_model_b", off_path_probs_b.get(Action.CALL, 0.0)],
        ["off_path_raise_prob_model_a", off_path_probs_a.get(Action.RAISE, 0.0)],
        ["off_path_raise_prob_model_b", off_path_probs_b.get(Action.RAISE, 0.0)],
    ]
    write_rows(out / "duhem_quine.csv", rows)


def merging(out: Path) -> None:
    hypotheses = default_hypotheses()
    truth = 2
    observations = sample_observations(hypotheses[truth].policy, 120, seed=11)
    friendly = BayesianOpponentModeler(hypotheses, np.array([0.2, 0.2, 0.2, 0.4]))
    dogmatic = BayesianOpponentModeler(hypotheses, np.array([0.4, 0.3, 0.0, 0.3]))
    rows = [["t", "friendly_truth_mass", "dogmatic_truth_mass"]]
    for t, (state, player, action) in enumerate(observations, start=1):
        friendly.update(state, player, action)
        dogmatic.update(state, player, action)
        rows.append([t, friendly.posterior[truth], dogmatic.posterior[truth]])
    write_rows(out / "merging.csv", rows)
    plot_lines(
        out / "merging.png",
        rows[1:],
        labels=["friendly truth mass", "dogmatic truth mass"],
        title="Merging of opinions",
        ylabel="posterior mass on truth",
    )


def popper_vs_bayes(out: Path) -> None:
    clean = ScriptedOpponent("clean_balanced", aggression=0.52, looseness=0.55, raise_bias=0.33, off_path_probe_response=0.5)
    noisy_truth = ScriptedOpponent("noisy_truth", aggression=0.58, looseness=0.6, raise_bias=0.33, off_path_probe_response=0.5)
    alternatives = [
        Hypothesis("clean", clean),
        Hypothesis("loose", ScriptedOpponent("loose", aggression=0.5, looseness=0.85, raise_bias=0.25, off_path_probe_response=0.5)),
        Hypothesis("aggro", ScriptedOpponent("aggro", aggression=0.85, looseness=0.55, raise_bias=0.65, off_path_probe_response=0.5)),
    ]
    observations = sample_observations(noisy_truth, 160, seed=23)
    bayes = BayesianOpponentModeler(alternatives)
    alive = np.ones(len(alternatives), dtype=bool)
    rows = [["t", "bayes_log_score", "popper_log_score", "alive_hypotheses"]]
    bayes_score = 0.0
    popper_score = 0.0
    for t, (state, player, action) in enumerate(observations, start=1):
        pred_bayes = sum(
            bayes.posterior[i] * alternatives[i].policy.action_probs(state, player).get(action, 0.0)
            for i in range(len(alternatives))
        )
        bayes_score += float(np.log(max(pred_bayes, 1e-15)))
        bayes.update(state, player, action)

        alive_indices = np.flatnonzero(alive)
        pred_popper = np.mean(
            [alternatives[i].policy.action_probs(state, player).get(action, 0.0) for i in alive_indices]
        )
        popper_score += float(np.log(max(pred_popper, 1e-15)))
        for i in alive_indices:
            if alternatives[i].policy.action_probs(state, player).get(action, 0.0) < 0.08:
                alive[i] = False
        if not alive.any():
            alive[:] = True
        rows.append([t, bayes_score, popper_score, int(alive.sum())])
    write_rows(out / "popper_vs_bayes.csv", rows)
    plot_lines(
        out / "popper_vs_bayes.png",
        rows[1:],
        labels=["Bayes cumulative log score", "Popper cumulative log score"],
        title="Popper vs Bayes under noisy likelihoods",
        ylabel="cumulative log score",
    )


def lakatos(out: Path) -> None:
    truth = ScriptedOpponent("truth", aggression=0.5, looseness=0.62, raise_bias=0.24, off_path_probe_response=0.82)
    train = sample_observations(truth, 120, seed=31)
    test = sample_observations(truth, 120, seed=37)
    programmes = [
        Hypothesis("core", ScriptedOpponent("core", aggression=0.5, looseness=0.55, raise_bias=0.24, off_path_probe_response=0.5), 3),
        Hypothesis("tilt_patch", ScriptedOpponent("tilt_patch", aggression=0.58, looseness=0.62, raise_bias=0.3, off_path_probe_response=0.5), 4),
        Hypothesis("off_path_patch", ScriptedOpponent("off_path_patch", aggression=0.5, looseness=0.62, raise_bias=0.24, off_path_probe_response=0.82), 4),
        Hypothesis("kitchen_sink", ScriptedOpponent("kitchen_sink", aggression=0.75, looseness=0.9, raise_bias=0.7, off_path_probe_response=0.92), 7),
    ]
    base_ll = predictive_log_likelihood(programmes[0].policy, test)
    rows = [["programme", "train_loglik", "test_loglik", "params", "test_gain_per_extra_param"]]
    for hyp in programmes:
        train_ll = predictive_log_likelihood(hyp.policy, train)
        test_ll = predictive_log_likelihood(hyp.policy, test)
        extra_params = max(hyp.parameter_count - programmes[0].parameter_count, 1)
        rows.append([hyp.name, train_ll, test_ll, hyp.parameter_count, (test_ll - base_ll) / extra_params])
    write_rows(out / "lakatos.csv", rows)


def dutch_book(out: Path) -> None:
    calibrated_call = 0.48
    miscalibrated_call = 0.68
    value_edge = 0.58
    rows = [["hand", "calibrated_bankroll", "miscalibrated_bankroll"]]
    calibrated = 0.0
    miscalibrated = 0.0
    for hand in range(1, 101):
        calibrated += calibrated_call * (2 * value_edge - 1)
        miscalibrated += miscalibrated_call * (2 * value_edge - 1)
        rows.append([hand, calibrated, miscalibrated])
    write_rows(out / "dutch_book.csv", rows)
    plot_lines(
        out / "dutch_book.png",
        rows[1:],
        labels=["calibrated opponent loss", "miscalibrated opponent loss"],
        title="Dutch book as repeated value betting",
        ylabel="expected chips transferred",
    )


def cfr_baseline(out: Path, iterations: int) -> None:
    solver = CFRSolver().train(iterations)
    avg = solver.average_strategy()
    rows = [
        ["metric", "value"],
        ["iterations", iterations],
        ["infosets", len(avg)],
        ["exploitability_chips_per_hand", exploitability(avg)],
    ]
    write_rows(out / "cfr_baseline.csv", rows)


def write_rows(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def plot_lines(path: Path, rows: list[list[object]], labels: list[str], title: str, ylabel: str) -> None:
    xs = [float(row[0]) for row in rows]
    plt.figure(figsize=(7, 4))
    for i, label in enumerate(labels, start=1):
        ys = [float(row[i]) for row in rows]
        plt.plot(xs, ys, label=label)
    plt.title(title)
    plt.xlabel("observation")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--cfr-iterations", type=int, default=100)
    args = parser.parse_args()
    run_all(args.out, cfr_iterations=args.cfr_iterations)


if __name__ == "__main__":
    main()

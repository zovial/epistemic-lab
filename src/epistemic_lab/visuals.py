from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon, Rectangle

from .leduc import Action, LeducState
from .models import BayesianOpponentModeler, Hypothesis, ScriptedOpponent
from .simulation import collect_decision_states, sample_observations

BG = "#08131f"
PANEL = "#101e2d"
GRID = "#253b55"
MUTED = "#8da1bd"
ORANGE = "#ffad3b"
CYAN = "#62ead9"
BLUE = "#89aefc"
WHITE = "#f3f7ff"
MAGENTA = "#ff6fba"


def epistemic_hypotheses() -> list[Hypothesis]:
    return [
        Hypothesis(
            "Race",
            ScriptedOpponent("Race", aggression=0.92, looseness=0.88, raise_bias=0.72, off_path_probe_response=0.45),
            parameter_count=4,
        ),
        Hypothesis(
            "Cautious",
            ScriptedOpponent("Cautious", aggression=0.16, looseness=0.22, raise_bias=0.1, off_path_probe_response=0.2),
            parameter_count=4,
        ),
        Hypothesis(
            "Conditional",
            ScriptedOpponent("Conditional", aggression=0.5, looseness=0.58, raise_bias=0.28, off_path_probe_response=0.9),
            parameter_count=4,
        ),
    ]


def generate_gallery(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    hypotheses = epistemic_hypotheses()
    observations = sample_observations(hypotheses[0].policy, 80, seed=91)
    flow = expected_log_likelihoods(hypotheses, truth_index=0)
    paths = [
        classroom_explainer(out / "classroom_explainer.png"),
        phase_portrait(hypotheses, flow, out / "phase_portrait_simplex.png"),
        thumbnail(hypotheses, flow, out / "thumbnail_epistemic_lab.png"),
        posterior_trace(hypotheses, observations, out / "posterior_trace.png"),
        off_path_probe(out / "off_path_probe.png"),
    ]
    return paths


def classroom_explainer(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(16, 9), facecolor="#f6f8fb")
    ax.set_facecolor("#f6f8fb")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.06, 0.9, "Epistemic Lab in one hand", fontsize=31, weight="bold", color="#142033")
    ax.text(
        0.06,
        0.85,
        "Poker turns evidence into a strategic signal: the action is chosen by someone who knows you are learning.",
        fontsize=15,
        color="#53657d",
    )

    boxes = [
        (0.06, 0.55, 0.18, 0.21, "1", "Hidden hand"),
        (0.30, 0.55, 0.18, 0.21, "2", "Strategic action"),
        (0.54, 0.55, 0.18, 0.21, "3", "Likelihood model"),
        (0.78, 0.55, 0.16, 0.21, "4", "Posterior"),
    ]
    for x, y, w, h, num, title in boxes:
        rounded_box(ax, x, y, w, h, "#ffffff", "#d8e0ea")
        ax.text(x + 0.018, y + h - 0.045, num, fontsize=24, weight="bold", color=ORANGE)
        ax.text(x + 0.058, y + h - 0.04, title, fontsize=15, weight="bold", color="#142033")
        if x < 0.76:
            arrow(ax, x + w + 0.025, y + h / 2, x + w + 0.07, y + h / 2)

    card(ax, 0.112, 0.595, "K", "#f1f4f8", "#101827", angle=-5)
    ax.text(0.105, 0.565, "private card", fontsize=12, color="#53657d")
    ax.text(0.35, 0.64, "BET", fontsize=27, weight="bold", color="#142033", ha="center")
    ax.text(0.35, 0.585, "observed signal", fontsize=12, color="#53657d", ha="center")
    ax.text(0.63, 0.64, "P(a | h, type)", fontsize=19, weight="bold", color="#142033", ha="center")
    ax.text(0.63, 0.585, "model the action", fontsize=12, color="#53657d", ha="center")

    posterior_bars(ax, 0.795, 0.595)

    rounded_box(ax, 0.11, 0.20, 0.34, 0.18, "#0f1d2d", "#223a55")
    ax.text(0.135, 0.315, "Passive observation", fontsize=18, weight="bold", color=WHITE)
    ax.text(0.135, 0.255, "Some opponent models look identical\nwhen you only watch normal hands.", fontsize=12.5, color=MUTED, linespacing=1.35)

    rounded_box(ax, 0.55, 0.20, 0.34, 0.18, "#0f1d2d", "#223a55")
    ax.text(0.575, 0.315, "Intervention", fontsize=18, weight="bold", color=WHITE)
    ax.text(0.575, 0.255, "Take a weird line, like a check-raise,\nand the models separate.", fontsize=12.5, color=MUTED, linespacing=1.35)

    arrow(ax, 0.47, 0.29, 0.53, 0.29, color=ORANGE, lw=3.0)
    ax.text(0.5, 0.15, "The lab makes philosophical claims testable with exact posteriors and CFR baselines.", ha="center", fontsize=15, color="#53657d")

    fig.savefig(path, dpi=190, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return path


def phase_portrait(
    hypotheses: list[Hypothesis],
    flow: np.ndarray,
    path: Path,
) -> Path:
    vertices = np.array([[0.08, 0.08], [0.50, 0.89], [0.92, 0.08]])
    rng = np.random.default_rng(4)
    priors = rng.dirichlet([0.62, 0.62, 0.62], size=230)
    trajectories = [bayes_flow_path(prior, flow, vertices, steps=90, strength=80) for prior in priors]

    fig, ax = plt.subplots(figsize=(10.5, 10.5), facecolor=BG)
    ax.set_facecolor(BG)
    draw_simplex(ax, vertices, title="PHASE PORTRAIT ON THE 2-SIMPLEX")
    add_trajectories(ax, trajectories, color=ORANGE, alpha=0.45, linewidth=0.85)
    ax.text(vertices[0, 0] - 0.03, vertices[0, 1] - 0.035, "RACE", color=ORANGE, fontsize=15, weight="bold")
    ax.text(vertices[1, 0] - 0.055, vertices[1, 1] + 0.035, "CAUTIOUS", color=CYAN, fontsize=15, weight="bold")
    ax.text(vertices[2, 0] - 0.075, vertices[2, 1] - 0.035, "CONDITIONAL", color=BLUE, fontsize=15, weight="bold")
    add_simplex_legend(ax)
    finish_axis(ax)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return path


def thumbnail(
    hypotheses: list[Hypothesis],
    flow: np.ndarray,
    path: Path,
) -> Path:
    vertices = np.array([[0.12, 0.16], [0.50, 0.84], [0.88, 0.16]])
    rng = np.random.default_rng(12)
    priors = rng.dirichlet([0.58, 0.58, 0.58], size=180)
    trajectories = [bayes_flow_path(prior, flow, vertices, steps=80, strength=78) for prior in priors]

    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
    ax.set_facecolor(BG)
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=BG, edgecolor="none"))
    ax.add_patch(Rectangle((0.015, 0.04), 0.97, 0.90, facecolor=PANEL, edgecolor="#203953", linewidth=1.4))
    ax.text(0.06, 0.86, "EPISTEMIC", color=WHITE, fontsize=36, weight="bold", family="DejaVu Sans Mono")
    ax.text(0.06, 0.775, "LAB", color=ORANGE, fontsize=68, weight="bold", family="DejaVu Sans Mono")
    ax.text(0.065, 0.705, "Strategic evidence in Leduc hold'em", color=MUTED, fontsize=17, family="DejaVu Sans Mono")
    ax.text(0.065, 0.165, "P(action | hand) is the unknown", color=CYAN, fontsize=17, family="DejaVu Sans Mono")

    card(ax, 0.067, 0.43, "J", "#f1f4f8", "#c2415d", angle=-8)
    card(ax, 0.145, 0.39, "Q", "#f1f4f8", "#1b365d", angle=5)
    card(ax, 0.222, 0.43, "K", "#f1f4f8", "#101827", angle=13)

    tri = vertices.copy()
    tri[:, 0] = 0.42 + 0.54 * tri[:, 0]
    tri[:, 1] = 0.09 + 0.82 * tri[:, 1]
    ax.add_patch(Polygon(tri, fill=False, edgecolor="#28415f", linewidth=1.6, alpha=0.9))
    shifted = []
    for traj in trajectories:
        t = traj.copy()
        t[:, 0] = 0.42 + 0.54 * t[:, 0]
        t[:, 1] = 0.09 + 0.82 * t[:, 1]
        shifted.append(t)
    add_trajectories(ax, shifted, color=ORANGE, alpha=0.52, linewidth=1.05)
    ax.scatter([tri[0, 0]], [tri[0, 1]], s=90, color=ORANGE, zorder=4)
    ax.text(tri[0, 0] - 0.025, tri[0, 1] - 0.07, "RACE", color=ORANGE, fontsize=19, weight="bold")
    ax.text(tri[1, 0] - 0.06, tri[1, 1] + 0.04, "CAUTIOUS", color=CYAN, fontsize=19, weight="bold")
    ax.text(tri[2, 0] - 0.09, tri[2, 1] - 0.07, "CONDITIONAL", color=BLUE, fontsize=19, weight="bold")
    ax.text(0.67, 0.91, "BAYES UNDER ADVERSARIAL EVIDENCE", color=MUTED, fontsize=16, family="DejaVu Sans Mono", ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return path


def posterior_trace(
    hypotheses: list[Hypothesis],
    observations: list[tuple[LeducState, int, Action]],
    path: Path,
) -> Path:
    modeler = BayesianOpponentModeler(hypotheses, np.array([0.2, 0.68, 0.12]))
    history = [modeler.posterior.copy()]
    for state, player, action in observations:
        history.append(modeler.update(state, player, action))
    arr = np.array(history)
    x = np.arange(len(arr))

    fig, ax = plt.subplots(figsize=(13, 7.2), facecolor=BG)
    ax.set_facecolor(BG)
    colors = [ORANGE, CYAN, BLUE]
    ax.stackplot(x, arr.T, colors=colors, alpha=0.78)
    for i, hyp in enumerate(hypotheses):
        ax.plot(x, arr[:, i], color=colors[i], linewidth=1.9, label=hyp.name)
    ax.set_title("POSTERIOR MASS DURING STRATEGIC EVIDENCE", color=WHITE, fontsize=20, pad=18, family="DejaVu Sans Mono")
    ax.set_xlabel("observed betting action", color=MUTED, fontsize=12)
    ax.set_ylabel("posterior probability", color=MUTED, fontsize=12)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(color=GRID, alpha=0.3, linewidth=0.8)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=12)
    for text in leg.get_texts():
        text.set_color(MUTED)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return path


def off_path_probe(path: Path) -> Path:
    model_a = ScriptedOpponent("on-path twin A", aggression=0.42, looseness=0.55, raise_bias=0.2, off_path_probe_response=0.15)
    model_b = ScriptedOpponent("on-path twin B", aggression=0.42, looseness=0.55, raise_bias=0.2, off_path_probe_response=0.9)
    state = (
        LeducState.initial()
        ._with_private(0, 4)
        .apply_action(Action.CHECK)
        .apply_action(Action.BET)
        .apply_action(Action.RAISE)
    )
    probs_a = model_a.action_probs(state, 1)
    probs_b = model_b.action_probs(state, 1)
    actions = list(state.legal_actions())
    x = np.arange(len(actions))
    width = 0.34

    fig, ax = plt.subplots(figsize=(11.5, 6.4), facecolor=BG)
    ax.set_facecolor(BG)
    ax.bar(x - width / 2, [probs_a[a] for a in actions], width, color=CYAN, alpha=0.82, label="twin A")
    ax.bar(x + width / 2, [probs_b[a] for a in actions], width, color=ORANGE, alpha=0.86, label="twin B")
    ax.set_title("OFF-PATH PROBE BREAKS OBSERVATIONAL EQUIVALENCE", color=WHITE, fontsize=18, pad=18, family="DejaVu Sans Mono")
    ax.set_xticks(x, [a.name for a in actions])
    ax.set_ylabel("P(action | check-raise line)", color=MUTED, fontsize=12)
    ax.tick_params(colors=MUTED)
    ax.set_ylim(0, 0.78)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(axis="y", color=GRID, alpha=0.35)
    leg = ax.legend(frameon=False)
    for text in leg.get_texts():
        text.set_color(MUTED)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return path


def expected_log_likelihoods(hypotheses: list[Hypothesis], truth_index: int) -> np.ndarray:
    states = collect_decision_states(player=1)
    scores = np.zeros(len(hypotheses), dtype=float)
    truth = hypotheses[truth_index].policy
    for state in states:
        truth_probs = truth.action_probs(state, 1)
        for i, hyp in enumerate(hypotheses):
            hyp_probs = hyp.policy.action_probs(state, 1)
            for action, q in truth_probs.items():
                scores[i] += q * np.log(max(hyp_probs.get(action, 0.0), 1e-15))
    scores /= len(states)
    return scores


def bayes_flow_path(
    prior: np.ndarray,
    expected_log_likelihood: np.ndarray,
    vertices: np.ndarray,
    steps: int,
    strength: float,
) -> np.ndarray:
    prior = np.clip(prior, 1e-9, 1.0)
    centered = expected_log_likelihood - expected_log_likelihood.max()
    points = []
    for t in np.linspace(0.0, 1.0, steps):
        logp = np.log(prior) + strength * t * centered
        weights = np.exp(logp - logp.max())
        posterior = weights / weights.sum()
        points.append(posterior @ vertices)
    return np.array(points)


def posterior_path(
    hypotheses: list[Hypothesis],
    prior: np.ndarray,
    observations: list[tuple[LeducState, int, Action]],
    vertices: np.ndarray,
) -> np.ndarray:
    modeler = BayesianOpponentModeler(hypotheses, prior)
    points = [prior @ vertices]
    for state, player, action in observations:
        points.append(modeler.update(state, player, action) @ vertices)
    return np.array(points)


def draw_simplex(ax: plt.Axes, vertices: np.ndarray, title: str) -> None:
    ax.add_patch(Polygon(vertices, fill=False, edgecolor="#25405f", linewidth=1.4, alpha=0.9))
    ax.text(0.03, 0.965, title, color=MUTED, fontsize=15, family="DejaVu Sans Mono", weight="bold", alpha=0.95)
    ax.plot([0.03, 0.97], [0.035, 0.035], color="#263b55", linewidth=0.8)


def add_trajectories(ax: plt.Axes, trajectories: list[np.ndarray], color: str, alpha: float, linewidth: float) -> None:
    segments = []
    for traj in trajectories:
        segments.extend(np.stack([traj[:-1], traj[1:]], axis=1))
    collection = LineCollection(segments, colors=color, linewidths=linewidth, alpha=alpha, capstyle="round")
    ax.add_collection(collection)


def add_simplex_legend(ax: plt.Axes) -> None:
    y = -0.005
    items = [(ORANGE, "converges to Race"), (CYAN, "converges to Cautious"), (BLUE, "converges to Conditional")]
    x = 0.03
    for color, label in items:
        ax.plot([x, x + 0.025], [y, y], color=color, linewidth=2.4, solid_capstyle="round", clip_on=False)
        ax.text(x + 0.04, y - 0.01, label, color=MUTED, fontsize=13, family="DejaVu Sans Mono", clip_on=False)
        x += 0.31


def card(ax: plt.Axes, x: float, y: float, rank: str, face: str, ink: str, angle: float) -> None:
    transform = (
        plt.matplotlib.transforms.Affine2D().rotate_deg_around(x + 0.035, y + 0.055, angle)
        + ax.transData
    )
    ax.add_patch(
        Rectangle((x, y), 0.07, 0.11, transform=transform, facecolor=face, edgecolor="#d8dee8", linewidth=1.0, zorder=5)
    )
    ax.text(x + 0.035, y + 0.056, rank, transform=transform, ha="center", va="center", color=ink, fontsize=24, weight="bold", zorder=6)


def rounded_box(ax: plt.Axes, x: float, y: float, w: float, h: float, face: str, edge: str) -> None:
    patch = plt.matplotlib.patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.3,
    )
    ax.add_patch(patch)


def arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float, color: str = "#aeb9c8", lw: float = 2.0) -> None:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={"arrowstyle": "->", "color": color, "lw": lw, "shrinkA": 0, "shrinkB": 0},
    )


def posterior_bars(ax: plt.Axes, x: float, y: float) -> None:
    labels = ["Race", "Cautious", "Cond."]
    values = [0.64, 0.12, 0.24]
    colors = [ORANGE, CYAN, BLUE]
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        yy = y + 0.075 - i * 0.048
        ax.text(x, yy, label, fontsize=10.5, color="#53657d", va="center")
        ax.add_patch(Rectangle((x + 0.06, yy - 0.012), 0.075, 0.024, facecolor="#e7edf5", edgecolor="none"))
        ax.add_patch(Rectangle((x + 0.06, yy - 0.012), 0.075 * value, 0.024, facecolor=color, edgecolor="none"))


def finish_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.04, 1.03)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/gallery"))
    args = parser.parse_args()
    for path in generate_gallery(args.out):
        print(path)


if __name__ == "__main__":
    main()

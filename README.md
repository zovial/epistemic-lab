# Epistemic Lab

Leduc hold'em as a small, exact testbed for Bayesian epistemology under strategic evidence.

The package includes:

- A two-player limit Leduc engine with exact chance enumeration.
- Vanilla CFR for a Nash-style baseline.
- Scripted opponent archetypes with known parameters.
- Exact and hierarchical Bayesian opponent modelers.
- Experiment harnesses for the five studies in the pitch.

## Quick start

Clone the repository, create a virtual environment, and install the package:

```bash
git clone https://github.com/zovial/epistemic-lab.git
cd epistemic-lab
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

Run the tests:

```bash
python3 -m pytest
```

## Experiments

Run the full experiment suite:

```bash
PYTHONPATH=src python3 -m epistemic_lab.experiments --out outputs/final_check
```

The harness writes CSV summaries and PNG plots to the output directory. The default run includes:

- Duhem-Quine on-path equivalence and off-path intervention diagnostics.
- Merging-of-opinions posterior trajectories.
- Popper-vs-Bayes noisy-signal comparison.
- Lakatos-style predictive log-likelihood-per-parameter scoring.
- Dutch-book expected value curve.
- CFR baseline exploitability summary.

For the long CFR oracle, run:

```bash
PYTHONPATH=src python3 -m epistemic_lab.experiments --out outputs/cfr_10k --cfr-iterations 10000
```

The default harness uses 100 CFR iterations so the full experiment suite stays quick.

## Visuals

Generate the visual gallery:

```bash
PYTHONPATH=src python3 -m epistemic_lab.visuals --out outputs/gallery
```

This produces thumbnail-style and figure-style PNGs, including:

- `outputs/gallery/classroom_explainer.png`
- `outputs/gallery/thumbnail_epistemic_lab.png`
- `outputs/gallery/phase_portrait_simplex.png`
- `outputs/gallery/posterior_trace.png`
- `outputs/gallery/off_path_probe.png`

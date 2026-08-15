# MO-PQUCB

Reference implementation for **Provably Efficient Personalized
Multi-Objective Bandits with Proactive Conversational Queries (UAI26)**.

MO-PQUCB learns objective rewards and user-specific preferences jointly. It
uses top-*m* rankings from proactive queries through a Plackett--Luce model,
anchors the shift-unidentifiable query estimate with bandit utility feedback,
and explores uncertainty in both rewards and preferences. `mo_pqucb_gl` uses a
group-lasso estimator for corrupted rankings.

The supported interface is the Python package in `src/` together with the
command-line scripts in `scripts/`. Historical notebooks, their generated
outputs, and machine-specific API settings are intentionally excluded from
the repository.

## Repository layout

```text
configs/                   experiment configurations
datasets/                  processed arrays used by real-world experiments
scripts/
  run_experiment.py        run one configured experiment
  run_sweep.py             run top-m or corruption-rate ablations
  generate_llm_rankings.py generate and checkpoint LLM rankings
  run_llm_experiment.py    replay cached rankings without API calls
src/mo_pqucb/
  config.py                configuration loading and validation
  environment.py           synthetic and real-world environments
  llm_pipeline.py          provider-neutral two-simulator LLM pipeline
  plackett_luce.py         online and robust PL estimators
  policies.py              MO-PQUCB and baseline policies
  runner.py                simulation, summaries, and plotting
tests/                     unit and end-to-end smoke tests
data/                      generated LLM caches (Git-ignored)
results/                   generated figures and summaries (Git-ignored)
```

## Installation

Python 3.9 or later is supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Install the optional provider SDKs only for the LLM experiment:

```bash
python -m pip install -e '.[llm]'
```

Equivalent dependency files are provided as `requirements.txt` and
`requirements-llm.txt`.

## Running experiments

Run all baselines and MO-PQUCB on the synthetic configuration:

```bash
python scripts/run_experiment.py --config configs/synthetic.json
```

Run a short experiment containing every baseline:

```bash
python scripts/run_experiment.py \
  --config configs/synthetic.json \
  --horizon 100 --runs 1 \
  --algorithms s_ucb s_moss pareto_ucb pareto_ts mo_oful conucb prucb mo_pqucb
```

Run the real-world experiments:

```bash
python scripts/run_experiment.py --config configs/tripadvisor.json
python scripts/run_experiment.py --config configs/beeradvocate.json
```

The runner keeps per-round arrays in memory. It writes plots directly as
`cumulative_regret.{pdf,png}` and, where applicable,
`preference_error.{pdf,png}`, together with `summary.json`. It does not save
`.npz` arrays. Curves show the mean and one standard error over independent
runs.

Run appendix ablations without changing a configuration file:

```bash
python scripts/run_sweep.py \
  --config configs/synthetic.json \
  --algorithms mo_pqucb \
  --top-m 1 5 10 20

python scripts/run_sweep.py \
  --config configs/synthetic_corrupt.json \
  --algorithms mo_pqucb_gl \
  --corruption-rates 0 0.1315633249 0.2 0.3 0.5
```

## Configuration

Each JSON file defines the random seed, horizon, number of runs, observed
ranking length (`top_m`), algorithms, environment, per-algorithm overrides,
and output directory. Important algorithm parameters are:

- `regularization`: ridge strength in preference estimation.
- `epsilon`: empirical confidence-radius multiplier.
- `reward_scale`: reward-UCB radius multiplier.
- `omega`: norm-weighted utility-update scale used by PRUCB/MO-PQUCB.
- `alpha`: bandit/query mixing weight; `null` selects the implementation's
  horizon-dependent default.
- `ucb_mode`: `empirical` reproduces the scale used by the experiments;
  `theoretical` enables the dimension-explicit theoretical radius.
- `response_feature`: MO-OFUL uses noisy observed rewards when set to
  `observed`; `mean` is retained for notebook diagnostics.
- `conversation_rate`: ConUCB's per-user logarithmic query-budget multiplier.
- `keyword_subset_size`: number of objectives defining each ConUCB keyword.
- `group_lasso_*`: robust-estimator penalty, optimization steps, and retained
  history for `mo_pqucb_gl`.

PRUCB, MO-OFUL, and ConUCB are separate implementations of their notebook
formulas. ConUCB uses one per-user query schedule: after `n_u` interactions
with user `u`, its budget is `conversation_rate * floor(log(n_u))`. Keyword
feedback is a scalar utility observation evaluated on the same keyword context
used by the regression update.

## LLM conversational experiment

The LLM workflow first generates and checkpoints natural-language queries and
inferred rankings, then evaluates the cached rankings offline. Set only the
credential for the selected provider:

```bash
export GROQ_API_KEY='...'
export GEMINI_API_KEY='...'
export DASHSCOPE_API_KEY='...'
```

Never place an API key in a source file or notebook. Supported provider names
are `groq`, `gemini`, and `dashscope`.

Example using Groq:

```bash
python scripts/generate_llm_rankings.py \
  --config configs/llm_tripadvisor.json \
  --provider groq \
  --model openai/gpt-oss-120b

python scripts/run_llm_experiment.py \
  --config configs/llm_tripadvisor.json \
  --cache-dir data/llm_cache/paper_llm_tripadvisor/openai_gpt-oss-120b
```

For a quick end-to-end smoke run, use matching overrides in both stages:

```bash
python scripts/generate_llm_rankings.py \
  --config configs/llm_tripadvisor.json \
  --provider groq --model openai/gpt-oss-120b \
  --horizon 100 --runs 1 --cache-dir data/llm_cache_smoke

python scripts/run_llm_experiment.py \
  --config configs/llm_tripadvisor.json \
  --cache-dir data/llm_cache_smoke/paper_llm_tripadvisor/openai_gpt-oss-120b \
  --horizon 100 --runs 1 --output-dir results/smoke
```

Generation is checkpointed after every batch and resumes automatically. The
pipeline requests structured output, validates batch sizes and permutations,
minimally repairs duplicate objective indices with a warning, and recursively
splits a ranking batch after persistent model failures. The LLM configuration
also controls `temperature`, `batch_size`, `rank_corruption`, and
`query_corruption`.

## Data

The repository includes only the processed arrays needed by the real-world
experiments:

- BeerAdvocate preferences `(14, 4)` and rewards `(14, 167, 4)`; rewards are
  averaged over users to construct the 167 arm feature vectors.
- TripAdvisor preferences `(10, 6)` and rewards `(62, 6)`.

Raw BeerAdvocate reviews and the TripAdvisor review corpus are omitted because
of their size. The algorithms do not require them unless the processed arrays
are regenerated.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The suite checks ranking validity, estimator identifiability, robust updates,
deterministic environments, LLM response parsing, configuration loading, and
a small complete run of every policy family.

## Implemented algorithms

- S-UCB and S-MOSS
- Pareto-UCB and Pareto-TS
- MO-OFUL
- ConUCB
- PRUCB
- MO-PQUCB and robust MO-PQUCB-GL

## Citation

```bibtex
@InProceedings{pmlr-v337-cao26a,
  title = {Provably Efficient Personalized Multi-Objective Bandits with Proactive Conversational Queries},
  author = {Cao, Linfeng and Shi, Ming and Shroff, Ness},
  booktitle = {Proceedings of the 42nd Conference on Uncertainty in Artificial Intelligence},
  pages = {900--942},
  year = {2026},
  volume = {337},
  publisher = {PMLR}
}
```

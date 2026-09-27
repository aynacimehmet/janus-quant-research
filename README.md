# janus-quant-research

**English** | [Türkçe](README.tr.md)

A research-oriented, end-of-day (EOD) quantitative portfolio engine. The
reference deployment targets Turkish mutual funds (TEFAS) and pension funds
(BES), with a ledger that models execution the same way in backtest and in
paper trading.

This is a sanitized public snapshot of an internal research system. It ships
**no market data**, **no personal configuration** and **no operational
records**. It is a local release candidate; nothing has been pushed to a
remote.

> Research software, not investment advice. The engine only produces a daily
> order **list**; orders are entered manually by a human.

## What is included

- `src/janus/` — the engine: data ingestion, feature store, quantile models,
  conformal calibration/selection, HRP allocation, ledger-aligned backtest,
  paper trading and reporting.
- `tests/` — the unit/integration suite, driven by deterministic synthetic
  generators.
- `config/janus.yaml` — example configuration (sanitized; fictional company
  names only).
- `docs/` — methodology contracts and a subset of Architecture Decision
  Records.

## Method highlights (implemented in this snapshot)

- **Point-in-time (as-of) data model.** A source-availability axis decides
  what a decision at day `D` may see; feature (`X`) and label (`y`) masks are
  separate; leakage is checked with a feature allow-list, a future-
  perturbation test and per-prediction model-version records.
- **Conformal selection.** CQR intervals plus an ACI-style daily `alpha`
  update; conformal p-values combined with a Benjamini-Hochberg FDR step
  (`q = 0.20`). Calibration and risk bands are separate.
- **Hierarchical Risk Parity (HRP).** Ledoit-Wolf covariance, fund / founder /
  cluster caps, and tax-aware rebalancing drift thresholds.
- **Ledger-aligned backtesting.** FIFO lots, date-dependent withholding,
  buy/sell valor and settlement, and distinct `can_buy` / `can_sell`,
  `data_stale` and `suspended` flags.
- **Validation.** Selection period vs. out-of-sample evaluation, 12 start
  dates as a sensitivity grid, purge/embargo walk-forward and CPCV, PBO/CSCV
  and deflated Sharpe in parameter sweeps, 5 seeds for stochastic modules.
- **Models.** Global LightGBM quantile models and a jump/HMM regime module.
- **Engineering.** Typer CLI, DuckDB/Parquet storage, MLflow experiment
  tracking, pytest + ruff + pre-commit.

## Out of scope (future work)

An adaptive exposure layer, including a possible reinforcement-learning
controller, is future research. No RL environment or agent is implemented in
this snapshot.

## Installation

```bash
conda env create -f environment.yml
conda activate janus
pip install -e ".[dev]"
cp .env.example .env   # fill in your own keys; never commit the local file
pre-commit install
janus doctor                  # environment / package / MPS / secret checks
```

## Usage

```bash
janus doctor
janus ingest --help
janus backtest
janus backtest-suite
janus gate-suite
```

## Testing and quality gates

```bash
pytest -q
ruff check src tests
pre-commit run --all-files
```

All tests run against synthetic data and require no market data or network
access.

## Data and privacy

- No market data, ledgers, personal paths, secrets or operational logs are
  included.
- Configuration uses fictional company names.
- Only synthetic fixtures under `tests/` and `sample_data/` are used.

## License

Released under the MIT License; see `LICENSE`. Third-party dependency
licenses are listed in `THIRD_PARTY_NOTICES.md`.

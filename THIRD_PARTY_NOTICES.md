# Third-party notices

This project depends on open-source packages. The project itself is released
under the MIT License (see `LICENSE`). The licenses below were verified from
the installed package metadata of the reference environment
(python 3.13, Apple silicon). They are all permissive and compatible with MIT.

## Core dependencies (verified from installed metadata)

| Package | License |
|---|---|
| pandas | BSD-3-Clause |
| numpy | BSD-3-Clause (bundled components: 0BSD, MIT, Zlib, CC0-1.0) |
| polars | MIT |
| duckdb | MIT |
| pyarrow | Apache-2.0 |
| borsapy | Apache-2.0 |
| requests | Apache-2.0 |
| PyYAML | MIT |
| pydantic-settings | MIT |
| typer | MIT |
| loguru | MIT |
| pandera | MIT |
| scipy | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| statsmodels | BSD-3-Clause |
| lightgbm | MIT |
| mlflow | Apache-2.0 |
| quantstats | Apache-2.0 |

## Optional extras (subset verified)

| Package | License |
|---|---|
| hmmlearn | BSD |
| jumpmodels | Apache-2.0 |
| yfinance | Apache-2.0 |

## Optional extras not installed in the reference environment

The following optional dependencies were not installed when this snapshot was
prepared, so their license metadata could not be verified locally. Confirm
them before enabling the corresponding extras:
`torch`, `gymnasium`, `catboost`, `mapie`, `alpaca-py`.

## Development dependencies (verified)

| Package | License |
|---|---|
| pytest | MIT |
| ruff | MIT |
| pre-commit | MIT |
| detect-secrets | Apache-2.0 (classifier; metadata License field is "UNKNOWN") |

No third-party source code is vendored in this repository.

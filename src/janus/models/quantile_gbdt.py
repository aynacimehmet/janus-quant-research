"""Global quantile LightGBM katmanı (S3b-2): FEATURE_COLUMNS beyaz listesi dışı girdi kabul edilmez.

TEMPORAL §5/§9: model yalnızca verilen eğitim satırlarını görür; preprocessing yok
(beyaz liste kolonları doğrudan girdi). Her α için ayrı quantile objective.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import pandas as pd

from janus.features.fund_features import validate_columns


class QuantileGBDT:
    """Her α için quantile objective'li GBDT; predict → q10/q50/q90 (alphas sırasına göre)."""

    def __init__(
        self,
        alphas: tuple[float, ...] = (0.1, 0.5, 0.9),
        num_leaves: int = 31,
        min_data_in_leaf: int = 1000,
        num_boost_round: int = 300,
        learning_rate: float = 0.05,
        seed: int = 0,
    ) -> None:
        self.alphas = tuple(sorted(alphas))
        self.num_leaves = int(num_leaves)
        self.min_data_in_leaf = int(min_data_in_leaf)
        self.num_boost_round = int(num_boost_round)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.models_: dict[float, lgb.Booster] = {}
        self.feature_names_: list[str] = []

    def _params(self, alpha: float) -> dict[str, Any]:
        return {
            "objective": "quantile",
            "alpha": alpha,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "deterministic": True,
            "force_col_wise": True,
            "verbose": -1,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> QuantileGBDT:
        validate_columns(X.columns)
        if len(X) == 0 or y.notna().sum() == 0:
            raise ValueError("eğitim için yeterli satır yok")
        self.feature_names_ = list(X.columns)
        ds = lgb.Dataset(X.to_numpy(dtype=float), label=y.to_numpy(dtype=float), feature_name=self.feature_names_)
        for a in self.alphas:
            self.models_[a] = lgb.train(self._params(a), ds, num_boost_round=self.num_boost_round)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        validate_columns(X.columns)
        if list(X.columns) != self.feature_names_:
            raise ValueError("tahmin kolonları eğitim kolonlarıyla aynı sıralamada olmalı")
        out = {f"q{int(round(a * 100))}": self.models_[a].predict(X.to_numpy(dtype=float)) for a in self.alphas}
        return pd.DataFrame(out, index=X.index, columns=[f"q{int(round(a * 100))}" for a in self.alphas])

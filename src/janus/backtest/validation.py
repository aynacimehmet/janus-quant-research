"""Walk-forward (anchored) bölmeleri: test 3 ay, purge 21 gün, embargo 5 gün; CPCV S2b'de."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_splits(
    calendar: pd.DatetimeIndex,
    test_months: int = 3,
    purge_days: int = 21,
    embargo_days: int = 5,
    min_train_days: int = 252,
    anchored: bool = True,
) -> list[Split]:
    """Test dilimleri takvim aylarına hizalı; train = [start, test_start − purge); sonraki train embargo kadar erken bitmez."""
    cal = calendar
    months = cal.to_period("M").unique()
    splits: list[Split] = []
    i = 0
    while i + test_months <= len(months):
        test_mask = cal.to_period("M").isin(months[i : i + test_months])
        test_idx = cal[test_mask]
        if len(test_idx) == 0:
            i += test_months
            continue
        t0 = cal.get_loc(test_idx[0])
        train_end_pos = t0 - purge_days - 1
        if train_end_pos - (0 if anchored else max(0, train_end_pos - 5 * 252)) >= min_train_days:
            splits.append(
                Split(
                    cal[0] if anchored else cal[max(0, train_end_pos - 5 * 252)],
                    cal[train_end_pos],
                    test_idx[0],
                    test_idx[-1],
                )
            )
        i += test_months
    return splits


def embargoed_train(calendar: pd.DatetimeIndex, split: Split, embargo_days: int = 5) -> pd.DatetimeIndex:
    """Bir sonraki eğitim penceresi için, önceki test bitişinden embargo kadar sonrasını dışlayan takvim."""
    end_pos = calendar.get_loc(split.test_end)
    cut = min(end_pos + embargo_days, len(calendar) - 1)
    return calendar[(calendar <= split.train_end) | (calendar > calendar[cut])]

from __future__ import annotations

import pandas as pd
from pandas.errors import MergeError

from ocpa.objects.log.importer.ocel2.sqlite import factory as ocel_import_factory


_ORIGINAL_DF_MERGE = pd.DataFrame.merge


def _safe_event_merge(self, right, *args, **kwargs):
    try:
        return _ORIGINAL_DF_MERGE(self, right, *args, **kwargs)
    except MergeError as e:
        msg = str(e)

        on = kwargs.get("on", None)
        how = kwargs.get("how", None)

        is_target_case = (
            isinstance(right, pd.DataFrame)
            and on == "event_id"
            and how == "left"
            and "duplicate columns" in msg
        )

        if not is_target_case:
            raise

        overlap = (set(self.columns) & set(right.columns)) - {"event_id"}

        related_suffix_cols = set()
        for c in overlap:
            related_suffix_cols.add(f"{c}_x")
            related_suffix_cols.add(f"{c}_y")

        cols_to_drop = [
            c for c in self.columns
            if c in overlap or c in related_suffix_cols
        ]

        if not cols_to_drop:
            raise

        left_clean = self.drop(columns=cols_to_drop, errors="ignore")

        print(
            "[my_ocel_importer] MergeError intercettato. "
            f"Ritento il merge rimuovendo da sinistra: {cols_to_drop}"
        )

        return _ORIGINAL_DF_MERGE(left_clean, right, *args, **kwargs)


def apply(file_path: str, parameters=None):
    if parameters is None:
        parameters = {}

    pd.DataFrame.merge = _safe_event_merge
    try:
        return ocel_import_factory.apply(file_path, parameters=parameters)
    finally:
        pd.DataFrame.merge = _ORIGINAL_DF_MERGE
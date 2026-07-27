"""Multi-table joining -- e.g. IEEE-CIS Fraud Detection's primary
`transaction` table plus its secondary `identity` table, joined on
`TransactionID`. Always a LEFT join from the primary table: most datasets of
this shape have only partial secondary-table coverage (e.g. most
transactions have no identity record), and losing primary rows that lack a
match would silently shrink the labeled dataset.
"""

from __future__ import annotations

import pandas as pd

from feature_generator.config import DatasetConfig
from feature_generator.profiling.profiler import read_table


def join_tables(config: DatasetConfig) -> pd.DataFrame:
    frames = {t.name: read_table(t.path) for t in config.tables}
    primary_cfg = next((t for t in config.tables if t.role == "primary"), None)
    if primary_cfg is None:
        raise ValueError(f"dataset '{config.name}' has no primary table")

    result = frames[primary_cfg.name]

    for table_cfg in config.tables:
        if table_cfg.role != "secondary":
            continue
        if not table_cfg.join_key:
            raise ValueError(f"secondary table '{table_cfg.name}' has no join_key configured")
        if table_cfg.join_key not in result.columns:
            raise ValueError(
                f"join_key '{table_cfg.join_key}' for secondary table '{table_cfg.name}' "
                f"is not a column of the primary table '{primary_cfg.name}'"
            )

        secondary = frames[table_cfg.name]
        overlapping = (set(result.columns) & set(secondary.columns)) - {table_cfg.join_key}
        if overlapping:
            secondary = secondary.rename(columns={c: f"{table_cfg.name}_{c}" for c in overlapping})

        result = result.merge(secondary, on=table_cfg.join_key, how="left")

    return result

"""Trusted negative fixture: deliberately malicious -- imports os and reads
the environment. Must be rejected by static_checks before Docker ever runs.
"""

import os

FEATURE_NAME = "malicious_env_leak"
REQUIRED_COLUMNS = ["Age"]


def fit(train_df, context):
    return {"secret": os.environ.get("HOME", "")}


def transform(df, params):
    return df["Age"] * 0 + len(params["secret"])

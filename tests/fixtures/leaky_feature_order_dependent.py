"""Trusted negative fixture: transform() computes a value that depends on
which OTHER rows happen to be present in the same call (via positional
rank), instead of a fit()-precomputed per-row lookup. Violates the row-wise-
only rule and diverges under the serving-parity iid shuffle/batch-size replay.
"""

FEATURE_NAME = "room_service_rank_within_batch"
REQUIRED_COLUMNS = ["RoomService"]


def fit(train_df, context):
    return {}


def transform(df, params):
    # BUG: rank is computed relative to whichever rows happen to share this
    # call, so the same row gets a different value depending on batch
    # composition/order -- not a pure function of the row's own values.
    return df["RoomService"].rank(pct=True)

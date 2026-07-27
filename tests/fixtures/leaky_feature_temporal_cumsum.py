"""Trusted negative fixture: transform() computes a running/cumulative
aggregate over whatever rows are present in the current call, instead of a
fit()-precomputed lookup keyed only by the row's own values. This is temporal
leakage: a row's value would be influenced by "future" rows visible in a
large offline batch that would not yet have arrived in real production
serving. Caught by the temporal replay's truncated-visibility check.
"""

FEATURE_NAME = "cumulative_room_service"
REQUIRED_COLUMNS = ["RoomService"]


def fit(train_df, context):
    return {}


def transform(df, params):
    # BUG: recomputed over whatever `df` happens to contain in this call,
    # rather than referencing a fit()-precomputed running total.
    return df["RoomService"].cumsum()

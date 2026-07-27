"""Trusted negative fixture: fit() memoizes the target encoding on its FIRST
call and reuses that memo verbatim for every subsequent fold, instead of
recomputing it fresh from whichever `train_df` it is actually given each
time. Because k-fold CV rotates which slice is held out, a later fold's
validation rows were part of an earlier fold's training split -- so the
memoized encoding leaks those very rows' own target values back into their
out-of-fold feature value. Requires `requires_target_in_fit=True` (the
target column must be present in `train_df` for this to run at all).
"""

_MEMO: dict | None = None

FEATURE_NAME = "leaky_category_target_mean"
REQUIRED_COLUMNS = ["category_id"]


def fit(train_df, context):
    global _MEMO
    # BUG: only computed once, on whichever fold's fit() call happens first;
    # every later call reuses it regardless of what `train_df` it receives.
    if _MEMO is None:
        _MEMO = train_df.groupby("category_id")[context.target_column].mean().to_dict()
    return {"means": _MEMO}


def transform(df, params):
    return df["category_id"].map(params["means"]).fillna(0.5)

"""Trusted negative-control fixture: a CORRECT feature module.

All aggregation happens once in fit() and is stored in params; transform()
only does a row-wise dict lookup. This is what every check in the safety
suite must PASS -- proving the checks aren't simply rejecting everything.
"""

FEATURE_NAME = "avg_room_service_by_home_planet"
REQUIRED_COLUMNS = ["HomePlanet", "RoomService"]


def fit(train_df, context):
    means = train_df.groupby("HomePlanet")["RoomService"].mean()
    global_mean = float(train_df["RoomService"].mean())
    return {"means": {str(k): float(v) for k, v in means.items()}, "global_mean": global_mean}


def transform(df, params):
    return df["HomePlanet"].map(params["means"]).fillna(params["global_mean"])

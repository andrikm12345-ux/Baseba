"""Tests for baseball feature engineering."""
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from src.data.features import (
    FEATURE_COLUMNS,
    build_features,
    build_inference_features,
    _form_winrate,
    _elo_update,
)
from collections import deque


def _make_games(n: int = 50, with_pitchers: bool = False) -> pd.DataFrame:
    """Generate synthetic MLB game data."""
    rng = np.random.default_rng(42)
    base = datetime(2024, 4, 1)
    records = []
    for i in range(n):
        home_id = rng.integers(1, 5)
        away_id = rng.integers(1, 5)
        while away_id == home_id:
            away_id = rng.integers(1, 5)
        home_runs = int(rng.poisson(4.3))
        away_runs = int(rng.poisson(4.3))
        row = {
            "id": i + 1,
            "utc_date": base + timedelta(days=i // 3),
            "home_team_id": int(home_id),
            "away_team_id": int(away_id),
            "home_runs": home_runs,
            "away_runs": away_runs,
            "status": "FINISHED",
            "competition": "mlb",
            "home_pitcher_era": float(rng.uniform(2.5, 6.0)) if with_pitchers else None,
            "home_pitcher_whip": float(rng.uniform(0.9, 1.6)) if with_pitchers else None,
            "home_pitcher_k9": float(rng.uniform(6.0, 12.0)) if with_pitchers else None,
            "home_pitcher_bb9": float(rng.uniform(1.5, 4.5)) if with_pitchers else None,
            "away_pitcher_era": float(rng.uniform(2.5, 6.0)) if with_pitchers else None,
            "away_pitcher_whip": float(rng.uniform(0.9, 1.6)) if with_pitchers else None,
            "away_pitcher_k9": float(rng.uniform(6.0, 12.0)) if with_pitchers else None,
            "away_pitcher_bb9": float(rng.uniform(1.5, 4.5)) if with_pitchers else None,
        }
        records.append(row)
    return pd.DataFrame(records)


def test_feature_columns_present():
    df = _make_games(60, with_pitchers=True)
    features = build_features(df)
    assert not features.empty
    for col in FEATURE_COLUMNS:
        assert col in features.columns, f"Missing feature: {col}"


def test_pitcher_features_fallback_to_avg():
    """Without pitcher data, ERA/WHIP should fall back to league averages."""
    from src.data.features import _ERA_AVG, _WHIP_AVG
    df = _make_games(30, with_pitchers=False)
    features = build_features(df)
    assert (features["home_pitcher_era"] == _ERA_AVG).all()
    assert (features["away_pitcher_era"] == _ERA_AVG).all()
    assert (features["pitcher_known"] == 0.0).all()


def test_pitcher_features_used_when_present():
    """With pitcher data, ERA should reflect actual values."""
    from src.data.features import _ERA_AVG
    df = _make_games(30, with_pitchers=True)
    features = build_features(df)
    # At least some ERA values should differ from the league average
    assert not (features["home_pitcher_era"] == _ERA_AVG).all()
    assert (features["pitcher_known"] == 1.0).all()


def test_era_diff_direction():
    """era_diff = home_era - away_era."""
    df = _make_games(30, with_pitchers=True)
    features = build_features(df)
    diff_check = features["home_pitcher_era"] - features["away_pitcher_era"]
    assert (abs(features["era_diff"] - diff_check) < 1e-6).all()


def test_target_columns_present():
    df = _make_games(60)
    features = build_features(df)
    for col in ("ml_home", "over85", "rl_home"):
        assert col in features.columns
        assert features[col].isin([0, 1]).all()


def test_no_data_leakage():
    """Feature at row i must only use data from rows < i."""
    df = _make_games(30)
    features = build_features(df)
    # First row should have default elo values (no prior games)
    assert abs(features.iloc[0]["home_elo"] - 1500.0) < 1.0
    assert abs(features.iloc[0]["away_elo"] - 1500.0) < 1.0


def test_inference_features_shape():
    df = _make_games(50, with_pitchers=True)
    finished = df[df["status"] == "FINISHED"].copy()
    upcoming = pd.DataFrame([{
        "id": 9999,
        "utc_date": datetime(2024, 6, 1),
        "home_team_id": 1,
        "away_team_id": 2,
        "home_runs": None,
        "away_runs": None,
        "status": "SCHEDULED",
        "competition": "mlb",
        "home_pitcher_era": 3.45,
        "home_pitcher_whip": 1.12,
        "home_pitcher_k9": 9.5,
        "home_pitcher_bb9": 2.8,
        "away_pitcher_era": 4.10,
        "away_pitcher_whip": 1.28,
        "away_pitcher_k9": 8.1,
        "away_pitcher_bb9": 3.2,
    }])
    inf = build_inference_features(upcoming, finished)
    assert len(inf) == 1
    assert "match_id" in inf.columns
    for col in FEATURE_COLUMNS:
        assert col in inf.columns, f"Missing inference feature: {col}"


def test_form_winrate():
    results = deque([(5, 3, "H"), (2, 4, "A"), (3, 1, "H")], maxlen=10)
    wr = _form_winrate(results)
    assert abs(wr - 2/3) < 1e-6


def test_elo_update_no_draw():
    h_elo, a_elo = _elo_update(1500, 1500, 5, 3)
    assert h_elo > 1500  # winner gains
    assert a_elo < 1500  # loser loses
    # No tie possible in MLB
    h_elo2, a_elo2 = _elo_update(1500, 1500, 0, 7)
    assert h_elo2 < 1500
    assert a_elo2 > 1500


def test_ml_home_target_correct():
    df = _make_games(40)
    features = build_features(df)
    merged = features.merge(
        df[["id", "home_runs", "away_runs"]].rename(columns={"id": "match_id"}),
        on="match_id"
    )
    for _, row in merged.iterrows():
        expected = int(row["home_runs"] > row["away_runs"])
        assert row["ml_home"] == expected


def test_over85_target_correct():
    df = _make_games(40)
    features = build_features(df)
    merged = features.merge(
        df[["id", "home_runs", "away_runs"]].rename(columns={"id": "match_id"}),
        on="match_id"
    )
    for _, row in merged.iterrows():
        expected = int(row["home_runs"] + row["away_runs"] > 8)
        assert row["over85"] == expected

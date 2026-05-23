"""Feature engineering for MLB baseball game prediction.

Builds, for every historical game, a feature vector available BEFORE first pitch
(no leakage): rolling form, run scoring/allowing rates, Elo rating, head-to-head.
No draws in baseball — all form/Elo uses binary win/loss.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "elo_diff",
    "home_elo",
    "away_elo",
    "home_win_rate",
    "away_win_rate",
    "home_rs_avg",
    "home_ra_avg",
    "away_rs_avg",
    "away_ra_avg",
    "home_rs_home_avg",
    "home_ra_home_avg",
    "away_rs_away_avg",
    "away_ra_away_avg",
    "h2h_home_winrate",
    "h2h_avg_runs",
    "h2h_home_avg_runs",
    "h2h_away_avg_runs",
    "h2h_recency_winrate",
    "rest_diff",
    # Starting pitcher features (None → filled with league averages)
    "home_pitcher_era",
    "away_pitcher_era",
    "era_diff",
    "home_pitcher_whip",
    "away_pitcher_whip",
    "whip_diff",
    "home_pitcher_k9",
    "away_pitcher_k9",
    "home_pitcher_bb9",
    "away_pitcher_bb9",
    "pitcher_known",  # 1 if both pitchers known, 0 otherwise
]

# League-average fallback values when pitcher is unknown
_ERA_AVG = 4.20
_WHIP_AVG = 1.30
_K9_AVG = 8.80
_BB9_AVG = 3.10


def _h2h_features(h2h_list: Deque[Tuple[int, int]], home_id: int, key_first_id: int) -> Dict[str, float]:
    if not h2h_list:
        return {
            "h2h_home_winrate": 0.5,
            "h2h_avg_runs": 8.5,
            "h2h_home_avg_runs": 4.3,
            "h2h_away_avg_runs": 4.2,
            "h2h_recency_winrate": 0.5,
        }
    home_is_first = home_id == key_first_id
    n = len(h2h_list)
    weights = list(range(1, n + 1))
    total_w = sum(weights)
    home_r: List[int] = []
    away_r: List[int] = []
    wins = 0
    weighted_wins = 0
    for w, (r0, r1) in zip(weights, h2h_list):
        hr, ar = (r0, r1) if home_is_first else (r1, r0)
        home_r.append(hr)
        away_r.append(ar)
        if hr > ar:
            wins += 1
            weighted_wins += w
    return {
        "h2h_home_winrate": wins / n,
        "h2h_avg_runs": float(np.mean([r0 + r1 for r0, r1 in h2h_list])),
        "h2h_home_avg_runs": float(np.mean(home_r)),
        "h2h_away_avg_runs": float(np.mean(away_r)),
        "h2h_recency_winrate": weighted_wins / total_w,
    }


@dataclass
class TeamState:
    elo: float = 1500.0
    last_results: Deque[Tuple[int, int, str]] = field(default_factory=lambda: deque(maxlen=10))  # (rs, ra, venue)
    last_home: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=10))
    last_away: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=10))
    last_game_date: pd.Timestamp | None = None


def _form_winrate(results: Deque[Tuple[int, int, str]]) -> float:
    """Win rate over last N games (no draws in baseball)."""
    if not results:
        return 0.5
    wins = sum(1 for rs, ra, _ in results if rs > ra)
    return wins / len(results)


def _avg(results: Deque, idx: int, default: float = 4.3) -> float:
    if not results:
        return default
    return float(np.mean([r[idx] for r in results]))


def _elo_expected(home_elo: float, away_elo: float, home_adv: float = 40.0) -> float:
    return 1.0 / (1.0 + 10 ** (-(home_elo + home_adv - away_elo) / 400.0))


def _elo_update(home_elo: float, away_elo: float, home_r: int, away_r: int, k: float = 20.0) -> Tuple[float, float]:
    expected_home = _elo_expected(home_elo, away_elo)
    score = 1.0 if home_r > away_r else 0.0  # no draws in MLB
    run_diff = abs(home_r - away_r)
    k_eff = k * (1 + np.log1p(run_diff) * 0.3)
    delta = k_eff * (score - expected_home)
    return home_elo + delta, away_elo - delta


def build_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Build features chronologically — state at row N reflects only rows < N.

    Expects columns: id, utc_date, home_team_id, away_team_id, home_runs, away_runs.
    Returns rows for games with known scores (training set) plus features.
    """
    df = matches_df.sort_values("utc_date").reset_index(drop=True).copy()

    team_state: Dict[int, TeamState] = defaultdict(TeamState)
    h2h: Dict[Tuple[int, int], Deque[Tuple[int, int]]] = defaultdict(lambda: deque(maxlen=10))

    feats: List[Dict] = []
    targets: List[Dict] = []

    for _, row in df.iterrows():
        home_id = int(row["home_team_id"])
        away_id = int(row["away_team_id"])
        h = team_state[home_id]
        a = team_state[away_id]

        rest_h = 1.0
        rest_a = 1.0
        if h.last_game_date is not None:
            rest_h = (row["utc_date"] - h.last_game_date).days
        if a.last_game_date is not None:
            rest_a = (row["utc_date"] - a.last_game_date).days

        key = tuple(sorted([home_id, away_id]))
        h2h_list = h2h[key]
        h2h_feats = _h2h_features(h2h_list, home_id, key[0])

        home_era = float(row["home_pitcher_era"]) if pd.notna(row.get("home_pitcher_era")) else _ERA_AVG
        away_era = float(row["away_pitcher_era"]) if pd.notna(row.get("away_pitcher_era")) else _ERA_AVG
        home_whip = float(row["home_pitcher_whip"]) if pd.notna(row.get("home_pitcher_whip")) else _WHIP_AVG
        away_whip = float(row["away_pitcher_whip"]) if pd.notna(row.get("away_pitcher_whip")) else _WHIP_AVG
        home_k9 = float(row["home_pitcher_k9"]) if pd.notna(row.get("home_pitcher_k9")) else _K9_AVG
        away_k9 = float(row["away_pitcher_k9"]) if pd.notna(row.get("away_pitcher_k9")) else _K9_AVG
        home_bb9 = float(row["home_pitcher_bb9"]) if pd.notna(row.get("home_pitcher_bb9")) else _BB9_AVG
        away_bb9 = float(row["away_pitcher_bb9"]) if pd.notna(row.get("away_pitcher_bb9")) else _BB9_AVG
        pitcher_known = float(
            pd.notna(row.get("home_pitcher_era")) and pd.notna(row.get("away_pitcher_era"))
        )

        feat = {
            "elo_diff": h.elo - a.elo,
            "home_elo": h.elo,
            "away_elo": a.elo,
            "home_win_rate": _form_winrate(h.last_results),
            "away_win_rate": _form_winrate(a.last_results),
            "home_rs_avg": _avg(h.last_results, 0, 4.3),
            "home_ra_avg": _avg(h.last_results, 1, 4.3),
            "away_rs_avg": _avg(a.last_results, 0, 4.3),
            "away_ra_avg": _avg(a.last_results, 1, 4.3),
            "home_rs_home_avg": _avg(h.last_home, 0, 4.5),
            "home_ra_home_avg": _avg(h.last_home, 1, 4.1),
            "away_rs_away_avg": _avg(a.last_away, 0, 4.1),
            "away_ra_away_avg": _avg(a.last_away, 1, 4.5),
            **h2h_feats,
            "rest_diff": rest_h - rest_a,
            "home_pitcher_era": home_era,
            "away_pitcher_era": away_era,
            "era_diff": home_era - away_era,
            "home_pitcher_whip": home_whip,
            "away_pitcher_whip": away_whip,
            "whip_diff": home_whip - away_whip,
            "home_pitcher_k9": home_k9,
            "away_pitcher_k9": away_k9,
            "home_pitcher_bb9": home_bb9,
            "away_pitcher_bb9": away_bb9,
            "pitcher_known": pitcher_known,
        }

        if pd.notna(row.get("home_runs")) and pd.notna(row.get("away_runs")):
            hr = int(row["home_runs"])
            ar = int(row["away_runs"])
            total_runs = hr + ar
            targets.append(
                {
                    "match_id": int(row["id"]),
                    "ml_home": int(hr > ar),            # moneyline: home wins
                    "over85": int(total_runs > 8),      # total runs > 8.5
                    "rl_home": int(hr - ar > 1),        # run line: home covers -1.5
                    "total_runs": total_runs,
                }
            )
            feats.append({"match_id": int(row["id"]), **feat})
            h.last_results.append((hr, ar, "H"))
            a.last_results.append((ar, hr, "A"))
            h.last_home.append((hr, ar))
            a.last_away.append((ar, hr))
            h.last_game_date = row["utc_date"]
            a.last_game_date = row["utc_date"]
            new_h_elo, new_a_elo = _elo_update(h.elo, a.elo, hr, ar)
            h.elo = new_h_elo
            a.elo = new_a_elo
            h2h[key].append((hr, ar) if home_id == key[0] else (ar, hr))

    if not feats:
        return pd.DataFrame(columns=["match_id", *FEATURE_COLUMNS, "ml_home", "over85", "rl_home"])
    f_df = pd.DataFrame(feats)
    t_df = pd.DataFrame(targets)
    return f_df.merge(t_df, on="match_id")


def build_inference_features(
    upcoming_df: pd.DataFrame, history_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute features for upcoming games by replaying history first."""
    all_df = pd.concat([history_df, upcoming_df], ignore_index=True).sort_values("utc_date").reset_index(drop=True)

    team_state: Dict[int, TeamState] = defaultdict(TeamState)
    h2h: Dict[Tuple[int, int], Deque[Tuple[int, int]]] = defaultdict(lambda: deque(maxlen=10))

    rows: List[Dict] = []
    upcoming_ids = set(upcoming_df["id"].astype(int).tolist())

    for _, row in all_df.iterrows():
        home_id = int(row["home_team_id"])
        away_id = int(row["away_team_id"])
        h = team_state[home_id]
        a = team_state[away_id]

        rest_h = 1.0 if h.last_game_date is None else (row["utc_date"] - h.last_game_date).days
        rest_a = 1.0 if a.last_game_date is None else (row["utc_date"] - a.last_game_date).days

        key = tuple(sorted([home_id, away_id]))
        h2h_list = h2h[key]
        h2h_feats = _h2h_features(h2h_list, home_id, key[0])

        home_era = float(row["home_pitcher_era"]) if pd.notna(row.get("home_pitcher_era")) else _ERA_AVG
        away_era = float(row["away_pitcher_era"]) if pd.notna(row.get("away_pitcher_era")) else _ERA_AVG
        home_whip = float(row["home_pitcher_whip"]) if pd.notna(row.get("home_pitcher_whip")) else _WHIP_AVG
        away_whip = float(row["away_pitcher_whip"]) if pd.notna(row.get("away_pitcher_whip")) else _WHIP_AVG
        home_k9 = float(row["home_pitcher_k9"]) if pd.notna(row.get("home_pitcher_k9")) else _K9_AVG
        away_k9 = float(row["away_pitcher_k9"]) if pd.notna(row.get("away_pitcher_k9")) else _K9_AVG
        home_bb9 = float(row["home_pitcher_bb9"]) if pd.notna(row.get("home_pitcher_bb9")) else _BB9_AVG
        away_bb9 = float(row["away_pitcher_bb9"]) if pd.notna(row.get("away_pitcher_bb9")) else _BB9_AVG
        pitcher_known = float(
            pd.notna(row.get("home_pitcher_era")) and pd.notna(row.get("away_pitcher_era"))
        )

        feat = {
            "match_id": int(row["id"]),
            "elo_diff": h.elo - a.elo,
            "home_elo": h.elo,
            "away_elo": a.elo,
            "home_win_rate": _form_winrate(h.last_results),
            "away_win_rate": _form_winrate(a.last_results),
            "home_rs_avg": _avg(h.last_results, 0, 4.3),
            "home_ra_avg": _avg(h.last_results, 1, 4.3),
            "away_rs_avg": _avg(a.last_results, 0, 4.3),
            "away_ra_avg": _avg(a.last_results, 1, 4.3),
            "home_rs_home_avg": _avg(h.last_home, 0, 4.5),
            "home_ra_home_avg": _avg(h.last_home, 1, 4.1),
            "away_rs_away_avg": _avg(a.last_away, 0, 4.1),
            "away_ra_away_avg": _avg(a.last_away, 1, 4.5),
            **h2h_feats,
            "rest_diff": rest_h - rest_a,
            "home_pitcher_era": home_era,
            "away_pitcher_era": away_era,
            "era_diff": home_era - away_era,
            "home_pitcher_whip": home_whip,
            "away_pitcher_whip": away_whip,
            "whip_diff": home_whip - away_whip,
            "home_pitcher_k9": home_k9,
            "away_pitcher_k9": away_k9,
            "home_pitcher_bb9": home_bb9,
            "away_pitcher_bb9": away_bb9,
            "pitcher_known": pitcher_known,
        }
        rows.append(feat)

        if pd.notna(row.get("home_runs")) and pd.notna(row.get("away_runs")):
            hr = int(row["home_runs"])
            ar = int(row["away_runs"])
            h.last_results.append((hr, ar, "H"))
            a.last_results.append((ar, hr, "A"))
            h.last_home.append((hr, ar))
            a.last_away.append((ar, hr))
            h.last_game_date = row["utc_date"]
            a.last_game_date = row["utc_date"]
            new_h_elo, new_a_elo = _elo_update(h.elo, a.elo, hr, ar)
            h.elo = new_h_elo
            a.elo = new_a_elo
            h2h[key].append((hr, ar) if home_id == key[0] else (ar, hr))

    feats_df = pd.DataFrame(rows)
    return feats_df[feats_df["match_id"].isin(upcoming_ids)].reset_index(drop=True)

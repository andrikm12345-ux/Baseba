"""Tests for baseball signal generation and settlement."""
import pandas as pd
import pytest

from src.signals.generator import Signal, _kelly, _make_signal, generate
from src.signals.tracker import _did_win


# ─── tracker tests ───────────────────────────────────────────────────────────

def test_ml_home_win():
    assert _did_win("ML", "HOME", 5, 3) is True
    assert _did_win("ML", "HOME", 3, 5) is False


def test_ml_away_win():
    assert _did_win("ML", "AWAY", 1, 7) is True
    assert _did_win("ML", "AWAY", 7, 1) is False


def test_total_over():
    # 8.5 line
    assert _did_win("TOTAL", "OVER", 5, 4) is True   # 9 > 8.5
    assert _did_win("TOTAL", "OVER", 4, 4) is False  # 8 < 8.5


def test_total_under():
    assert _did_win("TOTAL", "UNDER", 3, 4) is True   # 7 < 8.5
    assert _did_win("TOTAL", "UNDER", 5, 5) is False  # 10 > 8.5


def test_rl_cover():
    assert _did_win("RL", "COVER", 5, 3) is True    # 5-3=2 > 1.5
    assert _did_win("RL", "COVER", 4, 3) is False   # 4-3=1 < 1.5
    assert _did_win("RL", "COVER", 3, 4) is False   # loses outright


def test_rl_lay():
    assert _did_win("RL", "LAY", 2, 5) is True    # 5-2=3 > 1.5
    assert _did_win("RL", "LAY", 3, 4) is False   # 4-3=1 < 1.5
    assert _did_win("RL", "LAY", 5, 3) is False   # away loses


# ─── generator tests ─────────────────────────────────────────────────────────

def _make_row(**kwargs) -> pd.Series:
    defaults = {
        "match_id": 1,
        "p_home": 0.6,
        "p_away": 0.4,
        "p_over85": 0.55,
        "p_rl_home": 0.45,
        "odds_ml_home": 0.0,
        "odds_ml_away": 0.0,
        "odds_over85": 0.0,
        "odds_under85": 0.0,
        "odds_rl_home": 0.0,
        "odds_rl_away": 0.0,
        "_ai_applied": False,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


def test_generate_no_draw_market():
    """No DRAW or 1X2 signals should be generated."""
    row = _make_row(p_home=0.65, p_away=0.35)
    df = pd.DataFrame([row])
    signals = generate(df)
    markets = [s.market for s in signals]
    assert "1X2" not in markets
    assert "BTTS" not in markets
    assert "OU25" not in markets
    for m in markets:
        assert m in {"ML", "TOTAL", "RL"}


def test_generate_ml_picks_home():
    row = _make_row(p_home=0.70, p_away=0.30, odds_ml_home=1.85)
    df = pd.DataFrame([row])
    signals = generate(df)
    ml_signals = [s for s in signals if s.market == "ML"]
    assert ml_signals
    assert ml_signals[0].pick == "HOME"


def test_generate_ml_picks_away():
    row = _make_row(p_home=0.35, p_away=0.65, odds_ml_away=1.75)
    df = pd.DataFrame([row])
    signals = generate(df)
    ml_signals = [s for s in signals if s.market == "ML"]
    assert ml_signals
    assert ml_signals[0].pick == "AWAY"


def test_generate_value_signal_with_odds():
    """Edge = prob * odds - 1 >= MIN_EDGE."""
    row = _make_row(p_home=0.65, p_away=0.35, odds_ml_home=1.85)
    df = pd.DataFrame([row])
    signals = generate(df)
    ml = [s for s in signals if s.market == "ML" and s.pick == "HOME"]
    assert ml
    sig = ml[0]
    assert sig.is_value
    assert sig.edge == pytest.approx(0.65 * 1.85 - 1.0, abs=1e-3)


def test_kelly_stake_positive():
    stake = _kelly(0.6, 1.9)
    assert stake > 0
    assert stake <= 2.0


def test_kelly_stake_capped():
    stake = _kelly(0.9, 3.5)
    assert stake == 2.0


def test_confidence_gate():
    """Low confidence should produce no signal."""
    row = _make_row(p_home=0.51, p_away=0.49)
    df = pd.DataFrame([row])
    signals = generate(df)
    ml_model = [s for s in signals if s.market == "ML" and not s.is_value]
    assert not ml_model  # 0.51 < 0.60 model floor


def test_total_over_signal():
    row = _make_row(p_over85=0.65, p_home=0.55, p_away=0.45, odds_over85=1.80)
    df = pd.DataFrame([row])
    signals = generate(df)
    total = [s for s in signals if s.market == "TOTAL" and s.pick == "OVER"]
    assert total


def test_rl_cover_signal():
    row = _make_row(p_rl_home=0.62, p_home=0.55, p_away=0.45, odds_rl_home=1.90)
    df = pd.DataFrame([row])
    signals = generate(df)
    rl = [s for s in signals if s.market == "RL" and s.pick == "COVER"]
    assert rl

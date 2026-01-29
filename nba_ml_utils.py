"""Small, dependency-free helpers for tests and defensive population logic.
This mirrors the flattened `feat_*` -> attribute mapping used by backtest population code.
Keep this file light so tests can import it without loading heavy ML artifacts.
"""
from types import SimpleNamespace


def fv_from_flat_snapshot(snap: dict) -> SimpleNamespace:
    """Return a lightweight object with core attributes extracted from a flattened snapshot.

    Args:
        snap: mapping that may contain either raw keys (std, days_rest) or flattened keys (feat_std, feat_days_rest)
    """
    def g(k, default=None):
        return snap.get(k, snap.get(f'feat_{k}', default))

    std = g('std', None)
    try:
        std = float(std) if std is not None and std not in (-1.0, '') else None
    except Exception:
        std = None

    obj = SimpleNamespace(
        player_id=int(g('player_id', 0) or 0),
        player_name=g('player_name', '') or '',
        line=float(g('line', 0.0) or 0.0),
        avg_minutes=float(g('avg_minutes', 0.0) or 0.0),
        ema=float(g('ema', 0.0) or 0.0),
        std=std if std is not None else 1.0,
        opponent_drtg_season=float(g('opponent_drtg_season', 100.0) or 100.0),
        spread=float(g('spread', 0.0) or 0.0),
        game_total=float(g('game_total', 225.0) or 225.0),
        days_rest=int(g('days_rest', 2) or 2),
        is_home=bool(g('is_home', False)),
        is_b2b=bool(g('is_b2b', False)),
        games_played=int(g('games_played', 0) or 0),
        # common contextual features
        usage_rate=float(g('usage_rate', 0.0) or 0.0),
    )

    return obj

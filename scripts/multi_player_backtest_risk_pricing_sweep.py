"""Multi-player backtest sweep for Risk-Pricing thresholds

- Selects the top-N players from `data/ml_training_data.csv` (by frequency)
- Runs a walk-forward backtest per player (TEMPORAL INTEGRITY)
- Applies the TradingEngine grid and aggregates results across players
- Outputs:
  - data/multi_player_backtest_risk_pricing_sweep.csv (aggregated)
  - data/multi_player_backtest_risk_pricing_sweep_per_player.csv (per-player breakdown)

Defaults chosen for speed: top_n=25, test_days=30. Adjust CLI args for broader sweeps.
"""
from __future__ import annotations
import argparse
import itertools
import math
import os
from typing import List

import pandas as pd

from nba_prediction import PredictionOrchestrator
from trading_engine import TradingEngine

OUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'multi_player_backtest_risk_pricing_sweep.csv')
OUT_PER_PLAYER = os.path.join(os.path.dirname(__file__), '..', 'data', 'multi_player_backtest_risk_pricing_sweep_per_player.csv')
TRAINING_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'ml_training_data.csv')


def pick_top_players(training_csv: str, top_n: int = 25) -> List[str]:
    df = pd.read_csv(training_csv, usecols=[c for c in pd.read_csv(training_csv, nrows=1).columns if 'player' in c.lower() or 'name' in c.lower()])
    # heuristics for player name column
    name_cols = [c for c in df.columns if 'player' in c.lower() or 'name' in c.lower()]
    if not name_cols:
        raise SystemExit("Could not find player-name column in training CSV")
    col = name_cols[0]
    counts = pd.read_csv(training_csv, usecols=[col])[col].value_counts()
    return counts.head(top_n).index.tolist()


def _fv_from_snapshot(snap: dict):
    """Local helper: build a FeatureVector from a frozen snapshot dict (robust to legacy keys)."""
    from nba_prediction import FeatureVector
    def g(k, default=None):
        return snap.get(k, snap.get(f'feat_{k}', default))

    player_id = int(g('player_id', 0) or 0)
    player_name = g('player_name', '') or ''
    opponent_abbrev = g('opponent_abbrev', '') or ''
    market = g('market', 'PTS') or 'PTS'
    line = float(g('line', 0.0) or 0.0)
    avg_minutes = float(g('avg_minutes', 0.0) or 0.0)
    ema = float(g('ema', 0.0) or 0.0)
    std = g('std', None)
    std = float(std) if (std is not None and std not in (-1.0, '')) else 1.0
    opponent_drtg_season = float(g('opponent_drtg_season', 100.0) or 100.0)
    spread = float(g('spread', 0.0) or 0.0)
    game_total = float(g('game_total', 225.0) or 225.0)
    days_rest = int(g('days_rest', 2) or 2)
    is_home = bool(g('is_home', False))
    is_b2b = bool(g('is_b2b', False))
    games_played = int(g('games_played', 0) or 0)

    fv = FeatureVector(
        player_id=player_id,
        player_name=player_name,
        opponent_abbrev=opponent_abbrev,
        market=market,
        line=line,
        avg_minutes=avg_minutes,
        ema=ema,
        std=std,
        opponent_drtg_season=opponent_drtg_season,
        spread=spread,
        game_total=game_total,
        days_rest=days_rest,
        is_home=is_home,
        is_b2b=is_b2b,
        games_played=games_played,
    )

    for k, v in snap.items():
        if not k.startswith('feat_'):
            continue
        attr = k[len('feat_'):]
        if hasattr(fv, attr):
            try:
                setattr(fv, attr, v)
            except Exception:
                pass
    return fv


def run(top_n: int = 25, test_days: int = 30):
    orchestrator = PredictionOrchestrator()
    te = TradingEngine()

    players = pick_top_players(TRAINING_CSV, top_n)
    print(f"Running multi-player sweep for top {len(players)} players (example: {players[:3]})")

    grid = {
        'ev_threshold': [0.0, 0.01, 0.02, 0.03],
        'fractional_kelly': [0.10, 0.25, 0.50],
        'default_sigma': [3.0, 5.0],
        'sigma_multiplier': [1.0, 1.6],
        'z_score_threshold': [0.0, 0.5],
        'require_abs_z': [False, True]
    }
    combos = list(itertools.product(*(grid[k] for k in grid)))

    agg_rows = []
    per_player_rows = []
    bankroll = 10000.0

    for player in players:
        try:
            summary = orchestrator.run_backtest(player_name=player, market='PTS', lookback=15, test_days=test_days)
        except Exception as e:
            print(f"Backtest failed for {player}: {e}")
            continue
        if summary is None:
            print(f"No backtest results for {player}; skipping")
            continue

        base_df = summary.results_df.copy()
        # Reuse the single-player script's defensive population if needed
        if 'predicted_mean' not in base_df.columns or 'predicted_std' not in base_df.columns:
            me = orchestrator.model_engine
            rows = []
            for _, r in base_df.iterrows():
                feat = r.get('features') or {}
                try:
                    fv = _fv_from_snapshot(feat) if isinstance(feat, dict) else feat
                    reg = me.get_ml_regression_output(fv)
                    r['predicted_mean'] = reg.get('predicted_value')
                    r['predicted_std'] = reg.get('predicted_std')
                    r['predicted_ev'] = r.get('predicted_ev', reg.get('p_over'))
                    r['predicted_prob'] = r.get('predicted_prob', reg.get('p_over'))
                except Exception:
                    pass
                rows.append(r)
            base_df = pd.DataFrame(rows)

        # Normalize minimal columns expected by TradingEngine
        base_df['side'] = base_df.get('side', base_df.get('predicted_side', 'OVER'))
        base_df['odds'] = base_df.get('odds', 1.91)
        base_df['hit'] = base_df.get('hit', False)
        base_df['predicted_mean'] = pd.to_numeric(base_df.get('predicted_mean'), errors='coerce')
        base_df['predicted_std'] = pd.to_numeric(base_df.get('predicted_std'), errors='coerce')

        for combo in combos:
            params = dict(zip(grid.keys(), combo))
            try:
                processed = te.process_bets(
                    base_df,
                    bankroll=bankroll,
                    ev_threshold=params['ev_threshold'],
                    fractional_kelly=params['fractional_kelly'],
                    default_sigma=params['default_sigma'],
                    sigma_multiplier=params.get('sigma_multiplier', 1.0),
                    z_score_threshold=params['z_score_threshold'],
                    require_abs_z=params['require_abs_z']
                )
            except Exception as e:
                # record failure per-player
                per_player_rows.append({**params, 'player': player, 'selected': 0, 'net_profit': None, 'error': str(e)})
                continue

            if processed is None or len(processed) == 0:
                per_player_rows.append({**params, 'player': player, 'selected': 0, 'net_profit': 0.0, 'roi': 0.0, 'hit_rate': float('nan')})
                continue

            stake_col = 'rec_stake' if 'rec_stake' in processed.columns else ('recommended_stake' if 'recommended_stake' in processed.columns else None)
            if stake_col is None:
                per_player_rows.append({**params, 'player': player, 'selected': 0, 'net_profit': 0.0, 'roi': 0.0, 'hit_rate': float('nan')})
                continue

            odds_series = processed.get('odds', processed.get('odds_decimal', 1.91))
            profit = ((processed['hit'].astype(float) * (processed[stake_col] * (odds_series - 1))) - (~processed['hit'].astype(bool) * processed[stake_col]).astype(float)).sum()
            per_player_rows.append({**params, 'player': player, 'selected': len(processed), 'net_profit': float(profit), 'roi': float(profit) / bankroll, 'hit_rate': float(processed['hit'].mean()), 'avg_stake': float(processed[stake_col].mean())})

    # Aggregate across players
    if not per_player_rows:
        raise SystemExit('No per-player results collected; aborting')

    per_player_df = pd.DataFrame(per_player_rows)
    agg = per_player_df.groupby(list(grid.keys())).agg(
        players_with_bets=('player', lambda s: s.nunique()),
        total_selected=('selected', 'sum'),
        net_profit=('net_profit', 'sum'),
        mean_roi=('roi', 'mean'),
        mean_hit_rate=('hit_rate', 'mean')
    ).reset_index()

    agg = agg.sort_values(['mean_roi', 'net_profit'], ascending=False)
    agg.to_csv(OUT_CSV, index=False)
    per_player_df.to_csv(OUT_PER_PLAYER, index=False)

    print('\nTop 10 configs by mean ROI (aggregated across players):')
    print(agg.head(10).to_string(index=False, float_format='%.6f'))
    print(f"\nAggregated CSV: {OUT_CSV}\nPer-player CSV: {OUT_PER_PLAYER}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--top-n', type=int, default=25)
    ap.add_argument('--test-days', type=int, default=30)
    args = ap.parse_args()
    run(top_n=args.top_n, test_days=args.test_days)

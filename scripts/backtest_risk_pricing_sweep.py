"""Backtest sweep for Risk-Pricing thresholds

- Picks a representative player from `data/ml_training_data.csv` (most rows)
- Runs a walk-forward backtest (short window) using `PredictionOrchestrator`
- Feeds `BacktestSummary.results_df` into `TradingEngine.process_bets` across a small grid
- Produces CSV summary at `data/backtest_risk_pricing_sweep.csv` and prints top configs

Quick, reproducible and safe for local use.
"""
from __future__ import annotations
import itertools
import math
import os
import pandas as pd

from nba_prediction import PredictionOrchestrator
from trading_engine import TradingEngine

OUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'backtest_risk_pricing_sweep.csv')


def pick_representative_player(training_csv: str) -> str:
    # Read a small sample to discover player identifier columns (robust to large CSVs)
    sample = pd.read_csv(training_csv, nrows=2000)
    # common column names in various exports
    candidates = ['player_name', 'PLAYER_NAME', 'player', 'playerId', 'player_id', 'playerIdStr']
    for c in candidates:
        if c in sample.columns:
            return sample[c].value_counts().idxmax()

    # fallback: try watchlist.json
    watchlist_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'watchlist.json')
    try:
        wl = pd.read_json(watchlist_path)
        if 'player_name' in wl.columns:
            return wl['player_name'].iloc[0]
    except Exception:
        pass

    # last-resort hardcoded known player (should exist in most datasets)
    return 'LeBron James'


def run():
    orchestrator = PredictionOrchestrator()
    te = TradingEngine()

    # choose player
    training_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'ml_training_data.csv')
    player_name = pick_representative_player(training_csv)
    print(f"Selected player for sweep: {player_name}")

    # run a focused backtest (keep it small/reproducible)
    summary = orchestrator.run_backtest(player_name=player_name, market='PTS', lookback=15, test_days=60)
    if summary is None:
        raise SystemExit("Backtest returned no results for the chosen player")

    base_df = summary.results_df.copy()
    # Ensure necessary columns exist — if regression outputs are missing, call model_engine to populate
    required_cols = ['predicted_mean', 'predicted_std', 'predicted_ev', 'predicted_prob', 'line', 'date', 'market']
    missing = [c for c in required_cols if c not in base_df.columns]
    if missing:
        print(f"Backtest results missing columns {missing} — attempting to populate via ModelEngine...")
        from nba_prediction import FeatureVector
        me = orchestrator.model_engine

        def _fv_from_snapshot(snap: dict) -> FeatureVector:
            """Build a FeatureVector from a frozen snapshot dict (robust to legacy keys)."""
            # helper to read either 'attr' or 'feat_attr'
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

            # Populate any remaining feat_ fields present in the snapshot
            for k, v in snap.items():
                if not k.startswith('feat_'):
                    continue
                attr = k[len('feat_'):]
                if hasattr(fv, attr):
                    try:
                        setattr(fv, attr, v)
                    except Exception:
                        # ignore coercion failures for non-critical fields
                        pass
            return fv

        rows = []
        for _, r in base_df.iterrows():
            feat = r.get('features') or {}
            try:
                if isinstance(feat, FeatureVector):
                    fv = feat
                elif isinstance(feat, dict):
                    fv = _fv_from_snapshot(feat)
                else:
                    # attempt best-effort coercion
                    fv = _fv_from_snapshot(dict(feat))

                reg = me.get_ml_regression_output(fv)

                # align names used by TradingEngine (predicted_mean/predicted_std)
                r['predicted_mean'] = reg.get('predicted_value')
                r['predicted_std'] = reg.get('predicted_std')
                r['predicted_ev'] = r.get('predicted_ev', reg.get('p_over'))
                r['predicted_prob'] = r.get('predicted_prob', reg.get('p_over'))
            except Exception as e:
                # leave row as-is (will be filtered later)
                print(f"Failed to populate regression for row: {e}")
            rows.append(r)

        base_df = pd.DataFrame(rows)
        missing_after = [c for c in required_cols if c not in base_df.columns]
        if missing_after:
            raise SystemExit(f"Unable to populate required cols: {missing_after}")

    # --- Normalize backtest dataframe to TradingEngine expectations ---
    # Ensure there's a `side` column
    if 'side' not in base_df.columns:
        if 'predicted_side' in base_df.columns:
            base_df['side'] = base_df['predicted_side']
        elif 'recommended_side' in base_df.columns:
            base_df['side'] = base_df['recommended_side']
        else:
            base_df['side'] = base_df.apply(lambda r: 'OVER' if (r.get('predicted_mean') is not None and r.get('predicted_mean') > r.get('line', 0)) else 'UNDER', axis=1)

    # Ensure odds are present and in decimal form
    if 'odds' not in base_df.columns and 'odds_decimal' not in base_df.columns:
        base_df['odds'] = 1.91
    if 'odds_decimal' not in base_df.columns:
        base_df['odds_decimal'] = base_df.get('odds', 1.91).apply if hasattr(base_df.get('odds', 1.91), 'apply') else base_df.get('odds', 1.91)
        base_df['odds_decimal'] = base_df['odds'].astype(float).fillna(1.91)

    # Populate `hit` if missing (used for profit calc)
    if 'hit' not in base_df.columns:
        if 'actual_value' in base_df.columns:
            base_df['hit'] = ((base_df['side'] == 'OVER') & (base_df['actual_value'] > base_df['line'])) | \
                             ((base_df['side'] == 'UNDER') & (base_df['actual_value'] <= base_df['line']))
        else:
            base_df['hit'] = False

    # Coerce numeric prediction columns
    base_df['predicted_mean'] = pd.to_numeric(base_df.get('predicted_mean'), errors='coerce')
    base_df['predicted_std'] = pd.to_numeric(base_df.get('predicted_std'), errors='coerce')

    # Defensive: drop rows that have neither predicted_mean nor predicted_std
    base_df = base_df[~(base_df['predicted_mean'].isna() & base_df['predicted_std'].isna())].reset_index(drop=True)

    # Grid to sweep (kept deliberately small)
    grid = {
        'ev_threshold': [0.0, 0.01, 0.02, 0.03],
        'fractional_kelly': [0.10, 0.25, 0.50],
        'default_sigma': [3.0, 5.0],
        'z_score_threshold': [0.0, 0.5],
        'require_abs_z': [False, True]
    }

    combos = list(itertools.product(*(grid[k] for k in grid)))
    rows = []
    bankroll = 10000.0

    for combo in combos:
        params = dict(zip(grid.keys(), combo))
        try:
            processed = te.process_bets(
                base_df,
                bankroll=bankroll,
                ev_threshold=params['ev_threshold'],
                fractional_kelly=params['fractional_kelly'],
                default_sigma=params['default_sigma'],
                z_score_threshold=params['z_score_threshold'],
                require_abs_z=params['require_abs_z']
            )
        except Exception as e:
            print(f"Grid {params} failed: {e}")
            continue

        if processed is None or len(processed) == 0:
            # no bets selected under this policy
            rows.append({**params, 'selected': 0, 'net_profit': 0.0, 'roi': 0.0, 'hit_rate': float('nan'), 'avg_stake': 0.0})
            continue

        # compute simple metrics
        selected = len(processed)

        # stake column (TradingEngine v2 uses 'rec_stake')
        stake_col = 'rec_stake' if 'rec_stake' in processed.columns else ('recommended_stake' if 'recommended_stake' in processed.columns else None)
        if stake_col is None:
            print(f"Processed output missing stake column for params={params}; skipping")
            rows.append({**params, 'selected': 0, 'net_profit': 0.0, 'roi': 0.0, 'hit_rate': float('nan'), 'avg_stake': 0.0})
            continue

        odds_col = 'odds' if 'odds' in processed.columns else ('odds_decimal' if 'odds_decimal' in processed.columns else None)
        odds_series = processed[odds_col] if odds_col is not None else 1.91

        # profit per row: win -> stake*(odds-1); loss -> -stake
        profit = ((processed['hit'].astype(float) * (processed[stake_col] * (odds_series - 1))) - (~processed['hit'].astype(bool) * processed[stake_col]).astype(float)).sum()
        total_staked = processed[stake_col].sum()
        roi = float(profit) / bankroll if bankroll > 0 else 0.0
        avg_stake = float(processed[stake_col].mean())
        hit_rate = float(processed['hit'].mean()) if 'hit' in processed.columns else float('nan')

        rows.append({**params, 'selected': int(selected), 'net_profit': float(profit), 'roi': float(roi), 'hit_rate': float(hit_rate), 'avg_stake': float(avg_stake), 'total_staked': float(total_staked)})

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(['roi', 'net_profit'], ascending=False)
    out_df.to_csv(OUT_CSV, index=False)

    print('\nTop 5 configs by ROI:')
    print(out_df.head(5).to_string(index=False, float_format='%.6f'))
    print(f"\nFull results written to: {OUT_CSV}")


if __name__ == '__main__':
    run()

"""Ad-hoc backtest: promote Candidate 1 variance model and run Top-100, 30-day backtest.

Writes: data/promo_candidate1_backtest.csv and data/promo_candidate1_backtest_per_player.csv
"""
from __future__ import annotations
import os
import pandas as pd
from tqdm import tqdm

from nba_prediction import PredictionOrchestrator
from trading_engine import TradingEngine

BASE = os.path.dirname(__file__) or '.'
OUT_CSV = os.path.join(BASE, '..', 'data', 'promo_candidate1_backtest.csv')
OUT_PER = os.path.join(BASE, '..', 'data', 'promo_candidate1_backtest_per_player.csv')
TRAINING_CSV = os.path.join(BASE, '..', 'data', 'ml_training_data.csv')


def pick_top_players(training_csv: str, top_n: int = 100):
    df = pd.read_csv(training_csv, usecols=[c for c in pd.read_csv(training_csv, nrows=1).columns if 'player' in c.lower() or 'name' in c.lower()])
    name_cols = [c for c in df.columns if 'player' in c.lower() or 'name' in c.lower()]
    col = name_cols[0]
    counts = pd.read_csv(training_csv, usecols=[col])[col].value_counts()
    return counts.head(top_n).index.tolist()


def run(top_n: int = 100, test_days: int = 30, bankroll: float = 10000.0):
    orch = PredictionOrchestrator()
    te = TradingEngine()

    players = pick_top_players(TRAINING_CSV, top_n)
    rows = []
    per_player = []

    for player in tqdm(players, desc='players'):
        try:
            summary = orch.run_backtest(player_name=player, market='PTS', lookback=15, test_days=test_days)
        except Exception as e:
            per_player.append({'player': player, 'selected': 0, 'net_profit': None, 'error': str(e)})
            continue
        if summary is None:
            per_player.append({'player': player, 'selected': 0, 'net_profit': 0.0, 'roi': 0.0, 'hit_rate': float('nan')})
            continue

        base_df = summary.results_df.copy()
        # defensive population if needed (handle both `features` dict *and* flattened `feat_` columns)
        if 'predicted_mean' not in base_df.columns or 'predicted_std' not in base_df.columns:
            me = orch.model_engine

            # helper: build FeatureVector from either a snapshot dict or flattened feat_ columns
            from nba_prediction import FeatureVector
            def _fv_from_snapshot(snap: dict) -> FeatureVector:
                """Robust: accept legacy snapshot dicts or DataFrame rows with `feat_` prefixes."""
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

                # populate remaining feat_ fields if present
                for k, v in snap.items():
                    if not str(k).startswith('feat_'):
                        continue
                    attr = k[len('feat_'):]
                    if hasattr(fv, attr):
                        try:
                            setattr(fv, attr, v)
                        except Exception:
                            pass
                return fv

            rows = []
            for _, r in base_df.iterrows():
                feat = r.to_dict()  # works for both flattened rows and legacy dict in 'features'
                try:
                    # prefer the embedded 'features' dict when available
                    if 'features' in r and isinstance(r['features'], dict):
                        fv = _fv_from_snapshot(r['features'])
                    else:
                        fv = _fv_from_snapshot(feat)

                    reg = me.get_ml_regression_output(fv)

                    r['predicted_mean'] = reg.get('predicted_value')
                    r['predicted_std'] = reg.get('predicted_std')
                    r['predicted_ev'] = r.get('predicted_ev', reg.get('p_over'))
                    r['predicted_prob'] = r.get('predicted_prob', reg.get('p_over'))
                except Exception as e:
                    # keep original row (will be filtered later) but record the cause for debugging
                    r['__populate_error'] = str(e)
                rows.append(r)

            base_df = pd.DataFrame(rows)

        base_df['side'] = base_df.get('side', base_df.get('predicted_side', 'OVER'))
        base_df['odds'] = base_df.get('odds', 1.91)
        base_df['hit'] = base_df.get('hit', False)
        base_df['predicted_mean'] = pd.to_numeric(base_df.get('predicted_mean'), errors='coerce')
        base_df['predicted_std'] = pd.to_numeric(base_df.get('predicted_std'), errors='coerce')

        try:
            processed = te.process_bets(base_df, bankroll=bankroll, fractional_kelly=0.10, sigma_multiplier=1.2)
        except Exception as e:
            per_player.append({'player': player, 'selected': 0, 'net_profit': None, 'error': str(e)})
            continue

        if processed is None or len(processed) == 0:
            per_player.append({'player': player, 'selected': 0, 'net_profit': 0.0, 'roi': 0.0, 'hit_rate': float('nan')})
            continue

        stake_col = 'rec_stake'
        odds_series = processed.get('odds', processed.get('odds_decimal', 1.91))
        profit = ((processed['hit'].astype(float) * (processed[stake_col] * (odds_series - 1))) - (~processed['hit'].astype(bool) * processed[stake_col]).astype(float)).sum()
        per_player.append({'player': player, 'selected': len(processed), 'net_profit': float(profit), 'roi': float(profit) / bankroll, 'hit_rate': float(processed['hit'].mean()), 'avg_stake': float(processed[stake_col].mean())})

    per_df = pd.DataFrame(per_player)
    agg = per_df.agg({'player': 'count', 'selected': 'sum', 'net_profit': 'sum', 'roi': 'mean', 'hit_rate': 'mean'})
    agg_row = {
        'players_tested': int(per_df['player'].count()),
        'total_selected': int(per_df['selected'].sum()),
        'net_profit': float(per_df['net_profit'].sum()),
        'mean_roi': float(per_df['roi'].mean()),
        'mean_hit_rate': float(per_df['hit_rate'].mean())
    }
    pd.DataFrame([agg_row]).to_csv(OUT_CSV, index=False)
    per_df.to_csv(OUT_PER, index=False)
    return agg_row, per_df


if __name__ == '__main__':
    print('Running promoted-candidate backtest (Top-100, 30d)')
    agg, per = run(top_n=100, test_days=30, bankroll=10000.0)
    print('\nAggregate:')
    print(agg)
    print('\nPer-player (top 10 by net_profit):')
    print(per.sort_values('net_profit', ascending=False).head(10).to_string(index=False))

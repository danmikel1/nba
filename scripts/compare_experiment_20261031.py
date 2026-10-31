#!/usr/bin/env python3
"""
scripts/compare_experiment_20261031.py
=======================================
Grades the 14 KEPT bets vs the 6 SKIPPED bets from the 2026-10-31 slate
against official final ESPN game box scores and player stats to verify
whether our empirical playbook filtering increased win rate and profitability.

Usage:
    python scripts/compare_experiment_20261031.py
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.harvest_closing_odds import harvest_from_espn
from scripts.harvest_player_props import fetch_espn_player_boxscores


def _normalize_name(name: str) -> str:
    import unicodedata
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ASCII", "ignore").decode("utf-8")
    s = s.strip().lower()
    for ch in ("'", "’", "`", "."):
        s = s.replace(ch, "")
    s = s.replace("-", " ")
    for suffix in (" jr", " sr", " iii", " ii", " iv"):
        if s.endswith(suffix):
            s = s[:-len(suffix)].strip()
    return " ".join(s.split())


def grade_bet(bet: Dict[str, Any], espn_games: List[Dict[str, Any]], boxscores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    market = str(bet.get("market", "")).upper()
    is_game = market in ("SPREAD", "TOTAL") or bet.get("bet_type") in ("spread", "total")
    side = str(bet.get("side", "")).upper()
    line = float(bet.get("line", 0.0))

    graded = dict(bet)
    graded["status"] = "Pending"
    graded["actual"] = None
    graded["pnl"] = 0.0

    if is_game:
        # Match game by home and away
        # player field holds "AWAY @ HOME"
        player_str = bet.get("player", "")
        parts = [p.strip() for p in player_str.split("@")]
        if len(parts) != 2:
            return graded
        away_t, home_t = parts[0], parts[1]

        matching_game = None
        for g in espn_games:
            if g.get("home_team") == home_t and g.get("away_team") == away_t:
                matching_game = g
                break

        if not matching_game or matching_game.get("state") != "post":
            return graded

        h_score = matching_game.get("home_score", 0)
        a_score = matching_game.get("away_score", 0)
        act_margin = h_score - a_score
        act_total = h_score + a_score

        if market == "TOTAL":
            graded["actual"] = act_total
            if act_total == line:
                graded["status"] = "Push"
            elif "OVER" in side:
                graded["status"] = "Win" if act_total > line else "Loss"
            else:
                graded["status"] = "Win" if act_total < line else "Loss"

        elif market == "SPREAD":
            graded["actual"] = act_margin
            # Spread line logic: e.g. "MIN +12.5" -> MIN covered if margin or diff allows
            # If side is home team (e.g. NOP +4.5), NOP covers if act_margin > -4.5
            # If side is away team (e.g. MIN +12.5), MIN covers if act_margin < 12.5 (home wins by < 12.5)
            # Standard sportsbook check:
            is_home_pick = home_t in side
            pick_line = line if ("+" in side or line > 0) else -abs(line)
            if is_home_pick:
                cov = act_margin + pick_line
            else:
                cov = (-act_margin) + pick_line

            if cov == 0:
                graded["status"] = "Push"
            else:
                graded["status"] = "Win" if cov > 0 else "Loss"

    else:
        # Player prop
        p_name = _normalize_name(bet.get("player", ""))
        p_stats = None
        if p_name in boxscores:
            p_stats = boxscores[p_name]
        else:
            for b_k, b_v in boxscores.items():
                if p_name in b_k or b_k in p_name:
                    p_stats = b_v
                    break

        if not p_stats:
            return graded

        stat_val = p_stats.get(market)
        if stat_val is None:
            return graded

        graded["actual"] = float(stat_val)
        if stat_val == line:
            graded["status"] = "Push"
        elif "OVER" in side:
            graded["status"] = "Win" if stat_val > line else "Loss"
        else:
            graded["status"] = "Win" if stat_val < line else "Loss"

    # PnL calculation
    stake = float(bet.get("stake", 10.0) or 10.0)
    odds_am = int(bet.get("odds", -110) or -110)
    dec = 1.0 + (odds_am / 100.0) if odds_am > 0 else 1.0 + (100.0 / abs(odds_am))

    if graded["status"] == "Win":
        graded["pnl"] = stake * (dec - 1.0)
    elif graded["status"] == "Loss":
        graded["pnl"] = -stake
    else:
        graded["pnl"] = 0.0

    return graded


def main():
    exp_path = ROOT / "data" / "experiments" / "experiment_20261031_filtered_vs_skipped.json"
    if not exp_path.exists():
        print(f"Experiment file not found at {exp_path}")
        return

    exp = json.loads(exp_path.read_text(encoding="utf-8"))
    date_str = exp.get("date", "2026-10-31")

    print("=" * 84)
    print(f"📊 EXPERIMENT SETTLEMENT & HEAD-TO-HEAD AUDIT: {date_str}")
    print("=" * 84)

    # 1. Fetch live game and player stats from ESPN
    print(f"Fetching final scores & box scores for {date_str} from ESPN...")
    try:
        espn_games = harvest_from_espn(date_str)
        boxscores = fetch_espn_player_boxscores(date_str)
        print(f"  -> {len(espn_games)} games and {len(boxscores)} player box scores loaded.")
    except Exception as e:
        print(f"Error fetching ESPN data: {e}")
        espn_games, boxscores = [], {}

    # Grade both groups
    kept_graded = [grade_bet(b, espn_games, boxscores) for b in exp.get("kept_bets", [])]
    skipped_graded = [grade_bet(b, espn_games, boxscores) for b in exp.get("skipped_bets", [])]

    def _summarize_group(group_name: str, bets_list: List[Dict[str, Any]]):
        decided = [b for b in bets_list if b.get("status") in ("Win", "Loss")]
        pushes = [b for b in bets_list if b.get("status") == "Push"]
        pending = [b for b in bets_list if b.get("status") == "Pending"]
        wins = [b for b in decided if b.get("status") == "Win"]
        losses = [b for b in decided if b.get("status") == "Loss"]

        w = len(wins)
        l = len(losses)
        n = w + l
        wr = (w / n * 100.0) if n > 0 else 0.0
        pnl = sum(b.get("pnl", 0.0) for b in decided)

        print(f"\n--- {group_name.upper()} ({len(bets_list)} Bets) ---")
        print(f"Status: {w}W - {l}L - {len(pushes)}P ({len(pending)} pending)")
        if n > 0:
            print(f"Win Rate: {wr:.2f}% | Net PnL: ${pnl:+,.2f}")
        print(f"{'#':<3} | {'Market':<6} | {'Pick / Side':<16} | {'Line':>5} | {'Actual':>6} | {'Result':<7} | {'Edge / EV':<12}")
        print("-" * 72)
        for idx, b in enumerate(bets_list, 1):
            m = b.get("market", "")
            s = b.get("side", "")
            l_val = b.get("line", 0)
            act = b.get("actual")
            act_str = f"{act:.1f}" if act is not None else "-"
            res = b.get("status", "Pending")
            e_str = f"Edge {b['edge_pts']:.1f}" if b.get("edge_pts") is not None else f"EV {float(b.get('ev', 0))*100:+.1f}%"
            p_name = b.get("player", "")
            print(f"{idx:<3} | {m:<6} | {p_name[:12]} {s:<3} | {l_val:5.1f} | {act_str:>6} | {res:<7} | {e_str:<12}")

        return {"n": n, "w": w, "l": l, "wr": wr, "pnl": pnl, "pending": len(pending)}

    res_kept = _summarize_group("✅ KEPT PLAYS (Playbook-Compliant)", kept_graded)
    res_skipped = _summarize_group("❌ SKIPPED PLAYS (Noise / Traps)", skipped_graded)

    print("\n" + "=" * 84)
    print("🏆 FINAL EXPERIMENT VERDICT")
    print("=" * 84)
    if res_kept["n"] > 0 and res_skipped["n"] > 0:
        diff_wr = res_kept["wr"] - res_skipped["wr"]
        diff_pnl = res_kept["pnl"] - res_skipped["pnl"]
        print(f"Kept Plays Win Rate:    {res_kept['wr']:.1f}% ({res_kept['w']}W - {res_kept['l']}L)")
        print(f"Skipped Plays Win Rate: {res_skipped['wr']:.1f}% ({res_skipped['w']}W - {res_skipped['l']}L)")
        print(f"Win Rate Differential:  {diff_wr:+.1f}% in favor of Kept Plays")
        print(f"Profit Advantage:       ${diff_pnl:+,.2f} gained by filtering out skipped bets")
    elif res_kept["pending"] > 0 or res_skipped["pending"] > 0:
        print(f"Games / player props are still in progress or scheduled. Run again after final whistles to see full results!")


if __name__ == "__main__":
    main()

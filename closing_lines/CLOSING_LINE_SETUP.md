"""
CLOSING_LINE_SETUP.md — Integration Guide

You now have 3 tools to populate closing_line + clv fields without manual JSON editing.

=== QUICK START ===

1. FASTEST: CLI (No UI overhead)
   python closing_line_cli.py
   → Interactive prompt for each open bet. Type closing line, hit Enter.
   → Questions like: "[0] Luka Doncic PTS/OVER (Open→28.5): 28.5"
   → One keystroke per bet; ~50 bets in <2 mins if you rapid-fire.

2. BATCH: CSV Upload (50+ bets at once)
   1. Create CSV: player,market,line,side,closing_line
   2. Run: python closing_line_cli.py --csv lines.csv
   → Matches all rows to live bets, applies closing_line + calculates CLV.

3. STREAMLIT UI (Slower but visual)
   streamlit run closing_line_helper.py
   → Dropdown to pick bet, enter closing line, save.
   → Tab 2 has CSV batch upload for same 50-bet workflow.

=== OPTIONAL: TRY SCRAPER (1WIN & Asian Books) ===

If 1WIN/Bet88/BC Game are scrapable:
   python closing_line_scraper.py --book 1win --player "Luka Doncic" --market pts
   → May work; may not (sites block scrapers often).
   → If it breaks, stick to CLI/CSV.

Alternative: Sign up for BetRazor free tier:
   python closing_line_scraper.py --book betrazor --api-key YOUR_KEY --batch open_bets.json

=== WORKFLOW RECOMMENDATION ===

Daily flow (after your slate ends):

1. Let game results settle (~2-3 hours)
2. Open your sportsbook, note closing lines for each settled bet
3. SINGLE OPTION: Use CLI
   python closing_line_cli.py
   → Type each closing line as prompted. Takes 2-3 mins for 50 bets.
4. BATCH OPTION: Paste into CSV
   cat > today_lines.csv << EOF
   player,market,line,side,closing_line
   Luka Doncic,PTS,28.5,OVER,28.5
   Jayson Tatum,REB,8.5,OVER,9.0
   EOF
   python closing_line_cli.py --csv today_lines.csv

5. Check bet_tracker.json: closing_line + clv fields now populated
6. Next time nba_prediction.py runs, it uses real CLV instead of synthetic 0

=== CLV CALCULATION ===

CLV = closing_line - opening_line (for OVER)
CLV = -(closing_line - opening_line) (for UNDER)

Interpretation:
  CLV > 0   →  Market moved in your favor (sharp money agreement)
  CLV < 0   →  Market moved against you (sharp money disagreement)
  CLV = 0   →  Line remained static

Your model's job: predict games where you get +CLV, not zero CLV.
Once enough real CLV data exists, compare model signals to actual CLV → calibration feedback.

=== INTEGRATION: Closing Lines in analyze_performance.py ===

Already present in your code (line ~27):
  print(f"Total EV: {df['ev'].sum():.2f}")

After you populate bet_tracker with closing_line + clv:
  💡 Update analyze_performance.py to include CLV analysis:

    print(f"Avg CLV: {df['clv'].mean():.2f} pts")
    print(f"CLV Win%: {(df[df['clv'] > 0]['result']=='Win').mean():.1%}")
    print(f"Profitable CLV Bets: {(df['clv'] > 0) & (df['result']=='Win')}.sum()}")

This ties model edge (EV) to realized market edge (CLV).

=== ISSUES & TROUBLESHOOTING ===

Q: CLI not finding my bet?
A: bet_tracker.json may have typos (player name, market, date).
   Try: python -c "import json; d=json.load(open('data/bet_tracker.json')); print(d[0].keys())"
   Compare field names.

Q: Scraper returns no results?
A: 1WIN likely changed HTML or blocks Selenium.
   → Use CLI/CSV instead.
   → Consider higher-tier API (OddsPAPI, BetRazor paid tier).

Q: How do I know if closing_line was saved?
A: cat data/bet_tracker.json | grep -i closing_line (should show non-zero values, not 0)

Q: Can I retract a closing_line?
A: Edit bet_tracker.json manually, or run CLI again and re-enter.

=== NEXT STEPS ===

1. Test CLI with 1-2 bets: python closing_line_cli.py --bets 2
2. Once comfortable: Use daily after settling (2-3 mins for 50 bets)
3. After 50-100 closed bets with real CLV: analyze_performance.py will show CLV correlation
4. Update model gating: require not just EV > threshold, but CLV > 0 on backtests

Good luck! 🚀
"""

print(__doc__)

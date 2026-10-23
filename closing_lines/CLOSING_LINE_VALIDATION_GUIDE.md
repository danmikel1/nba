"""
CLOSING_LINE_VALIDATION_GUIDE.md

Lines are constantly moving. This guide helps you:
1. Record closing lines correctly (without typos)
2. Validate lines against realistic market movement
3. Detect stale/inaccurate data
4. Track patterns to build intuition

## The Problem

Your current data shows:
❌ ALL closing lines = opening lines (0 movement)
❌ This is UNREALISTIC for prop markets

Real markets:
✓ PTS props: typically move ±0.5-2.0 pts
✓ REB/AST props: typically move ±0.3-1.5 pts
✓ PA props: typically move ±0.5-2.0 pts
✓ ZERO movement happens only if:
  • Market consensus agrees with you (rare)
  • Line was captured at EXACT moment bet placed (very rare)
  • Data is synthetic/fake

## Solution: 3-Tool Validation Pipeline

### Step 1: WHILE Recording → Use Validation Checklist
python closing_line_validation_checklist.py

• Interactive mode: guides you through each bet
• Detects common errors:
  - Typos (28.5 → 285)
  - Swap errors (OVER/UNDER reversed)
  - Stale lines (recorded too late)
  - Copy-paste errors
• Shows context from recent similar bets
• Confidence level: HIGH / MEDIUM / LOW

Usage:
  python closing_line_validation_checklist.py --player "Luka" --market PTS --opening 28.5 --closing 29.0 --side OVER

### Step 2: AFTER Recording → Use Validator Summary
python closing_line_validator.py

• Full report on all closing lines
• Detects patterns:
  - % of bets with zero movement (flags synthetic data)
  - Extreme movers (|move| > 2.0 pts) - may need review
  - Movement by market (PTS vs REB vs AST patterns)
  - Duplicate suspicious patterns
• Recommends capture windows (when to record)

Usage:
  python closing_line_validator.py              # Full analysis
  python closing_line_validator.py --check-typos   # Focus on errors
  python closing_line_validator.py --windows    # Show capture time recommendations

### Step 3: BUILD INTUITION → Use Movement Tracker
python line_movement_tracker.py

• Understand what's NORMAL:
  - Expected movement by market
  - Movement range (5th-95th percentile)
  - OVER vs UNDER bias
  - Daily trends
• Correlation check:
  - Do lines that move your way (+CLV) result in more wins?
  - If no, your closing lines may be inaccurate
• Time series: see if recent lines are more realistic

Usage:
  python line_movement_tracker.py           # All stats
  python line_movement_tracker.py --by-market  # By market type
  python line_movement_tracker.py --correlation  # CLV → Win correlation

## Recommended Workflow

### Daily (After settling):

1. Open your sportsbook, note closing lines
2. Use CLI to record:
   python closing_line_cli.py
   
3. For each bet, also validate (optional but recommended):
   python closing_line_validation_checklist.py --player "X" --market PTS --opening 28.5 --closing 29.0 --side OVER

4. When done recording, run full report:
   python closing_line_validator.py

5. If you see RED FLAGS from report → fix those bets

### Weekly:

Run movement tracker to spot trends:
python line_movement_tracker.py

If movement stats look off (still mostly zero), you know data quality is poor.

## When Lines Should Move (Realistic Expectations)

### PTS (Points)
- Typical:    ±0.5 to ±1.5 pts
- Extreme:    ±2.5 to ±3.0 pts only on injury/news
- Zero move:  Very rare, <5% of the time

### REB / AST (Rebounds / Assists)
- Typical:    ±0.3 to ±1.0 pt
- Extreme:    ±1.5 to ±2.0 pts only on depth changes
- Zero move:  <10% of the time

### PA / PRA (Combined Props)
- Typical:    ±0.5 to ±2.0 pts
- Extreme:    ±2.5 to ±3.5 pts
- Zero move:  <10% of the time

### Movement Direction Pattern
- OVER bets: If you got a good price, sharps may push line UP (positive CLV)
- UNDER bets: If you got a good price, sharps may push line DOWN (negative CLV)
- If consistent: Market agreed with your bets (good sign)
- If opposite: Market disagreed (market was sharper)

## Red Flags in Your Data

### 🚨 CRITICAL (Fix Immediately)
- [X] All closing_line = opening_line (movement = 0.00)
  → Data is synthetic or you're copying opening lines
  → Action: Record REAL closing lines from your sportsbook

### ⚠️ WARNING (Investigate)
- Closing line far from opening (>3.0 pts) without news
  → Typo? Or real sharps disagreement?
- All closing lines at .5 or .0 (rounded)
  → Suggests guessing/rounding instead of actual lines
- Same % of +CLV as -CLV (should be ~50%)
  → Indicates no real edge, or market is random

### ✓ GOOD (Keep doing)
- ~20-40% zero movement (market didn't move much)
- ~30-40% positive movement (sharps agreed)
- ~20-40% negative movement (market disagreed)
- Most movement < ±2.0 pts

## FAQ

Q: Why do ALL my lines show zero movement?
A: You're either:
   1. Recording synthetic/fake closing lines (for backtesting)
   2. Copying opening lines instead of actual closing lines
   3. Using fake/test data
   Fix: When you record closing lines, use your REAL sportsbook's line AT SETTLEMENT.

Q: When should I record closing lines?
A: Ideally 5-30 mins after game start. This captures:
   • Market reaction to initial action
   • Before sharp money fully moves in
   • Clean closing line (bet is settled)

Q: What if I record too late?
A: Lines may move 2-3+ pts by 2-3 hours later.
   This isn't "wrong" data, but it's stale.
   Impact: CLV was probably higher when you originally got the line.

Q: Can I validate historical bets?
A: Yes, but requires historical line data (expensive/paid APIs).
   Instead: Trust validator logic, spot-check a few manually.

Q: Should all my lines move?
A: No. ~20-30% static is normal (market consensus, no new info).
   But >80% static = suspicious.

## Integration with Other Tools

- analyze_performance.py: Now shows CLV analysis if closing lines exist
- nba_prediction.py: Uses real CLV in backtests (instead of synthetic 0)
- Your model: Can learn from CLV patterns (which factors predict +CLV?)

## Next Steps After Validation

1. Make sure 50+ bets have real closing lines (not synthetic)
2. Run: python closing_line_validator.py --check-typos (fix obvious errors)
3. Run: python line_movement_tracker.py --correlation (check CLV quality)
4. If movement variance is low (<0.5 pts), re-examine data sources
5. Once data quality is high, use CLV feedback to improve model

Good luck! 🚀
"""

print(__doc__)

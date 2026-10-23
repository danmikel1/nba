import json
from datetime import date, timedelta

TRACKER = "data/bet_tracker.json"
TIER1_THRESHOLD = 0.045
cutoff = (date.today() - timedelta(days=2)).isoformat()

with open(TRACKER, encoding="utf-8") as f:
    bets = json.load(f)

in_window = [b for b in bets if b.get("date", "") >= cutoff]
print(f"Cutoff date : {cutoff}")
print(f"Bets in window: {len(in_window)}")

flagged = []
for b in in_window:
    qs = float(b.get("quality_score", 0) or 0)
    stake = float(b.get("stake", 0) or 0)
    if stake > 0 and qs <= TIER1_THRESHOLD:
        flagged.append(b)

if not flagged:
    print("No rejected bets with stale stakes found.")
else:
    print(f"\nRejected (qs <= {TIER1_THRESHOLD}) but stake > 0 — zeroing {len(flagged)} bet(s):\n")
    for b in flagged:
        print(f"  [{b['date']}] {b.get('player')} | qs={b.get('quality_score'):.4f} | stake={b.get('stake'):.2f} | result={b.get('result')}")
        b["stake"] = 0.0

    with open(TRACKER, "w", encoding="utf-8") as f:
        json.dump(bets, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Zeroed {len(flagged)} stake(s) and saved bet_tracker.json")

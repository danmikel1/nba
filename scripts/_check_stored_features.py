import json

bets = json.loads(open('data/bet_tracker.json', encoding='utf-8').read())
sample = next((b for b in bets if b.get('result') in ['Win', 'Loss']), None)
if sample:
    feats = {k: v for k, v in sample.items() if k.startswith('feat_')}
    print(f"Bet: {sample.get('player')} {sample.get('date')}")
    print(f"feat_ keys count: {len(feats)}")
    print(f"feat_miles_96h: {feats.get('feat_miles_96h', 'NOT STORED')}")
    print(f"feat_altitude_next: {feats.get('feat_altitude_next', 'NOT STORED')}")
    print()
    print("All feat_ keys:")
    for k, v in feats.items():
        print(f"  {k}: {v}")

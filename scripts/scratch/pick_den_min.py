import json

# Load depth charts
with open('data/depth_charts.json', encoding='utf-8') as f:
    depth_charts = json.load(f)

# Build player to team mapping
player_to_team = {}
for team, positions in depth_charts.items():
    if team == '_timestamp':
        continue
    for position, players in positions.items():
        for player in players:
            if player not in player_to_team:
                player_to_team[player] = team

# Load bets
with open('data/bet_tracker.json', encoding='utf-8') as f:
    bets = json.load(f)

# Get today's DEN vs MIN bets
today_bets = [b for b in bets if b.get('date') == '2026-04-30']
den_min_bets = [b for b in today_bets if 
                (player_to_team.get(b.get('player')) == 'DEN' and b.get('opponent') == 'MIN') or
                (player_to_team.get(b.get('player')) == 'MIN' and b.get('opponent') == 'DEN')]

# Sort by EV descending
den_min_bets.sort(key=lambda x: x.get('ev', 0), reverse=True)

print("DEN vs MIN - All Bets Sorted by EV:")
print("="*80)
for i, bet in enumerate(den_min_bets, 1):
    player = bet.get('player')
    market = bet.get('market')
    line = bet.get('line')
    side = bet.get('side')
    ev = bet.get('ev')
    proj = bet.get('proj')
    player_team = player_to_team.get(player)
    opponent = bet.get('opponent')
    
    marker = " <-- CURRENT" if i == 1 else ""
    print(f"{i}. {player} ({player_team}) vs {opponent}")
    print(f"   {market} {line} {side} | EV: {ev:.4f} | Proj: {proj:.2f}{marker}")
    print()

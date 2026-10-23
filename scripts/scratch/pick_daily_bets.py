import json
from collections import defaultdict

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

# Get today's bets
today_bets = [b for b in bets if b.get('date') == '2026-04-30']

# Find matchups by identifying which teams play each other
matchups = set()
for bet in today_bets:
    player = bet.get('player')
    opponent = bet.get('opponent')
    player_team = player_to_team.get(player)
    if player_team and opponent:
        # Create matchup as sorted tuple
        matchup = tuple(sorted([player_team, opponent]))
        matchups.add(matchup)

print("Today's Matchups (3 games):")
matchups = sorted(matchups)
for i, (team1, team2) in enumerate(matchups, 1):
    print(f'{i}. {team1} vs {team2}')
    
# Now pick the best bet for each matchup
print("\n" + "="*80)
print("BEST BET FOR EACH MATCHUP (by EV):")
print("="*80 + "\n")

for team1, team2 in matchups:
    # Get all bets from team1 playing team2
    matchup_bets = [b for b in today_bets if 
                    (player_to_team.get(b.get('player')) == team1 and b.get('opponent') == team2) or
                    (player_to_team.get(b.get('player')) == team2 and b.get('opponent') == team1)]
    
    if matchup_bets:
        best = max(matchup_bets, key=lambda x: x.get('ev', 0))
        player = best.get('player')
        market = best.get('market')
        line = best.get('line')
        side = best.get('side')
        ev = best.get('ev')
        proj = best.get('proj')
        player_team = player_to_team.get(player)
        opponent = best.get('opponent')
        
        print(f"{team1} vs {team2}")
        print(f"  Pick: {player} ({player_team}) vs {opponent}")
        print(f"  Bet: {market} {line} {side}")
        print(f"  EV: {ev:.4f} | Projection: {proj:.2f}")
        print()

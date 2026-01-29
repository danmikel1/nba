import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nba_prediction import DataLoader, CONFIG

print('Testing fetch_player_positions...')
loader = DataLoader(CONFIG)
pos_map = loader.fetch_player_positions(['2025-26','2024-25'], player_ids=[254,201939], force_refresh=True)
print('Fetched positions count:', len(pos_map))
for k,v in list(pos_map.items())[:20]:
    print(k, v)

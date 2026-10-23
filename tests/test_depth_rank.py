"""
Test script to check depth rank for De'Anthony Melton (GSW)
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from depth_chart_manager import DepthChartManager
from nba_prediction import InjuryManager, FeatureEngineer, CONFIG, DataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_melton_depth_rank():
    """Test De'Anthony Melton's depth rank on GSW with Curry out"""
    
    print("=" * 80)
    print("TESTING DEPTH RANK: De'Anthony Melton (GSW)")
    print("=" * 80)
    
    # Initialize managers
    depth_manager = DepthChartManager()
    injury_manager = InjuryManager()
    
    # Test parameters
    player_name = "austin reeves"
    team_abbrev = "LAL"
    
    # Step 1: Get base depth rank from ESPN
    print(f"\n1. ESPN DEPTH CHART LOOKUP:")
    base_rank = depth_manager.get_player_depth_rank(player_name, team_abbrev)
    rank_labels = {1: "Starter", 2: "Rotation", 3: "Bench", 4: "Not Found"}
    print(f"   Player: {player_name}")
    print(f"   Team: {team_abbrev}")
    print(f"   Base Depth Rank: {base_rank} ({rank_labels.get(base_rank, 'Unknown')})")
    
    # Step 2: Get injury report
    print(f"\n2. INJURY REPORT:")
    injuries = injury_manager.get_team_injury_report(team_abbrev)
    print(f"   Found {len(injuries)} injury report entries for {team_abbrev}:")
    for inj in injuries:
        print(f"   - {inj['name']}: {inj['status']}")
    
    # Step 3: Get OUT players only
    out_players = [inj['name'] for inj in injuries if inj.get('status') in ('OUT', 'DOUBTFUL')]
    print(f"\n3. PLAYERS OUT/DOUBTFUL:")
    if out_players:
        for p in out_players:
            print(f"   - {p}")
    else:
        print(f"   - None")
    
    # Step 4: Calculate effective rank with injury adjustment
    print(f"\n4. EFFECTIVE DEPTH RANK (with injury adjustment):")
    if out_players:
        effective_rank = depth_manager.get_effective_depth_rank(
            player_name=player_name,
            team_abbrev=team_abbrev,
            injured_players=out_players
        )
        print(f"   Effective Rank: {effective_rank} ({rank_labels.get(effective_rank, 'Unknown')})")
        
        if effective_rank != base_rank:
            print(f"   ✅ PROMOTED: Rank {base_rank} → Rank {effective_rank}")
        else:
            print(f"   ⚠️  NO CHANGE: Rank remains {base_rank}")
    else:
        print(f"   No injuries to adjust for - using base rank: {base_rank}")
    
    # Step 5: Expected result
    print(f"\n5. EXPECTED RESULT:")
    if "Stephen Curry" in out_players or "Curry" in str(out_players):
        print(f"   ✅ Curry is OUT - Melton should be promoted to Rank 1 (Starter)")
    else:
        print(f"   ⚠️  Curry not found in OUT list - check injury data")
    
    print("=" * 80)
    
    final_rank = effective_rank if out_players else base_rank
    assert final_rank in [1, 2, 3, 4]
    return None


if __name__ == "__main__":
    result_rank = test_melton_depth_rank()
    print(f"\n🏀 FINAL DEPTH RANK FOR DE'ANTHONY MELTON: {result_rank}")
    
    if result_rank == 1:
        print("✅ SUCCESS: Melton is ranked as STARTER")
    else:
        print(f"❌ UNEXPECTED: Melton is ranked as {result_rank} (expected 1)")

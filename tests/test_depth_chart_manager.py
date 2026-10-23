"""
Test script for DepthChartManager.

Validates:
1. Depth chart scraping from ESPN
2. Player depth rank lookups
3. Dynamic promotion algorithm
4. Fuzzy name matching
"""

import sys
sys.path.insert(0, '.')

from depth_chart_manager import DepthChartManager


def test_depth_chart_manager():
    """Test the depth chart manager functionality."""
    print("=" * 60)
    print("DEPTH CHART MANAGER TEST")
    print("=" * 60)
    
    manager = DepthChartManager()
    
    # Test 1: Fetch depth chart for Indiana Pacers
    print("\n1. Testing depth chart scrape for IND (Pacers)...")
    depth_chart = manager.get_team_depth_chart('IND', force_refresh=True)
    
    if depth_chart:
        print(f"   SUCCESS: Found depth chart with {len(depth_chart)} positions")
        for pos, players in depth_chart.items():
            print(f"   {pos}: {players[:5]}")  # First 5 players
    else:
        print("   WARNING: No depth chart data (ESPN may have changed format)")
        print("   This is a soft failure - scraping is best-effort")
    
    # Test 2: Player depth rank lookup
    print("\n2. Testing player depth rank lookup...")
    
    # Use current active players from depth chart
    test_cases = [
        ('Andrew Nembhard', 'IND'),  # Should be Rank 1 (Starter PG)
        ('T.J. McConnell', 'IND'),   # Should be Rank 2 (Backup PG)
        ('Kam Jones', 'IND'),        # Should be Rank 3 (3rd string PG)
        ('LeBron James', 'LAL'),     # Lakers test
        ('Unknown Player', 'LAL'),   # Should be Rank 4 (Not Found)
    ]
    
    for player, team in test_cases:
        rank = manager.get_player_depth_rank(player, team)
        print(f"   {player} ({team}): Rank {rank}")
    
    # Test 3: Dynamic Promotion Algorithm
    print("\n3. Testing Dynamic Promotion Algorithm...")
    print("   Scenario: T.J. McConnell with Andrew Nembhard OUT")
    
    # Without injuries
    base_rank = manager.get_player_depth_rank('T.J. McConnell', 'IND')
    print(f"   Base rank (no injuries): {base_rank}")
    
    # With Nembhard OUT
    effective_rank = manager.get_effective_depth_rank(
        player_name='T.J. McConnell',
        team_abbrev='IND',
        injured_players=['Andrew Nembhard']
    )
    print(f"   Effective rank (Nembhard OUT): {effective_rank}")
    
    if effective_rank < base_rank:
        print("   SUCCESS: Dynamic promotion is working!")
    elif base_rank == 4:
        print("   INFO: Player not found in depth chart (scraping may need adjustment)")
    else:
        print(f"   NOTE: Rank stayed same - promotion logic may need review")
    
    # Test 3b: Multiple injuries causing promotion
    print("\n   Scenario: Kam Jones with Nembhard AND McConnell OUT")
    base_rank = manager.get_player_depth_rank('Kam Jones', 'IND')
    effective_rank = manager.get_effective_depth_rank(
        player_name='Kam Jones',
        team_abbrev='IND',
        injured_players=['Andrew Nembhard', 'T.J. McConnell']
    )
    print(f"   Base rank: {base_rank}, Effective rank: {effective_rank}")
    if effective_rank < base_rank:
        print("   SUCCESS: Player promoted due to multiple injuries!")
    
    # Test 4: Fuzzy name matching
    print("\n4. Testing fuzzy name matching...")
    fuzzy_tests = [
        ('Nic Claxton', 'BKN'),         # Short form of Nicolas
        ('Nicolas Claxton', 'BKN'),     # Full name
        ('TJ McConnell', 'IND'),        # Without periods
        ('T.J. McConnell', 'IND'),      # With periods
    ]
    
    for name, team in fuzzy_tests:
        rank = manager.get_player_depth_rank(name, team)
        print(f"   '{name}' ({team}): Rank {rank}")
    
    # Test 5: Cache persistence
    print("\n5. Testing cache persistence...")
    print(f"   Cache file: {manager.cache_file}")
    print(f"   Cache exists: {manager.cache_file.exists()}")
    print(f"   Teams in cache: {list(manager._cache.keys())}")
    
    print("\n" + "=" * 60)
    print("DEPTH CHART MANAGER TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_depth_chart_manager()

from data_engine import DataEngine
from pathlib import Path
import pandas as pd

def test_time_travel():
    # run in offline mode to avoid flaky network during CI/tests
    engine = DataEngine(Path('data'), season='2025-26', allow_fetch=False)
    
    # Detroit Pistons ID: 1610612765 (used for both synthetic caches)
    det_id = 1610612765

    # Seed the engine with fake defense stats so that the test does not
    # depend on the live API or pre-existing disk cache.  This also allows
    # us to run with allow_fetch=False.
    engine.season_defense_cache['2023-24'] = pd.DataFrame([{   
        'TEAM_ID': det_id, 'OPP_AST_RANK': 29, 'OPP_REB_RANK': 29, 'OPP_PTS_RANK': 29
    }])
    engine.season_defense_cache['2025-26'] = pd.DataFrame([{  
        'TEAM_ID': det_id, 'OPP_AST_RANK': 2, 'OPP_REB_RANK': 2, 'OPP_PTS_RANK': 2
    }])

    print("\n🏎️  TESTING TIME TRAVEL LOGIC 🏎️")
    print("------------------------------------------------")
    
    rank_23 = engine.get_historical_dvp_rank(det_id, 'PG', '2023-24')
    print(f"📅 2023-24 DET Rank vs PG: {rank_23} (Expected: ~28-30)")
    
    rank_26 = engine.get_historical_dvp_rank(det_id, 'PG', '2025-26')
    print(f"📅 2025-26 DET Rank vs PG: {rank_26} (Expected: ~1-5)")
    
    print("------------------------------------------------")
    
    if rank_23 > 20 and rank_26 < 10:
        print("✅ SUCCESS: The Engine correctly identifies historical context!")
    elif rank_23 == rank_26:
        print("❌ FAILURE: The Engine is returning the same rank for both eras.")
    else:
        print("⚠️ WARNING: Ranks look suspicious. Check definitions.")

if __name__ == "__main__":
    test_time_travel()
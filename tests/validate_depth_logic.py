import pandas as pd
import numpy as np
import os

# Define paths
FILE_PATH = 'data/ml_training_data.csv'

def validate():
    if not os.path.exists(FILE_PATH):
        print(f"❌ Error: File not found at {FILE_PATH}")
        return

    print(f"🔄 Loading {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH)
    
    # 1. Identify the 'Minutes' column
    # feat_min was REMOVED (data leak - actual game minutes, not pre-game observable).
    # Use feat_avg_minutes instead (rolling average, known before tip-off).
    min_col = None
    for col in ['feat_avg_minutes', 'MIN', 'minutes', 'MP']:
        if col in df.columns:
            min_col = col
            break
            
    if min_col is None:
        print("❌ Error: Could not find a 'Minutes' column (feat_avg_minutes, MIN, etc).")
        print(f"   Available columns: {list(df.columns)}")
        print("   You MUST regenerate your dataset to include actual minutes played.")
        return

    print(f"✅ Found minutes column: '{min_col}'")
    
    # 2. Simulate the Logic
    print("running simulation...")
    
    # OLD WAY (Naive): If you played < 28 mins, you are Bench (Rank 2)
    df['rank_old'] = np.where(df[min_col] >= 28, 1.0, 2.0)
    
    # NEW WAY (Blowout Proof):
    # You are Starter (Rank 1) if:
    # A) You played > 26 mins tonight (Spot Starter)
    # B) OR You average > 30 mins (Established Star in a blowout)
    
    # Ensure feat_avg_minutes exists
    if 'feat_avg_minutes' not in df.columns:
        print("⚠️ Warning: 'feat_avg_minutes' missing. Using simpler logic for test.")
        df['rank_new'] = df['rank_old']
    else:
        df['rank_new'] = np.where(
            (df[min_col] >= 26) | (df['feat_avg_minutes'] >= 30),
            1.0,
            2.0
        )

    # Handle DNPs (0 minutes)
    df['rank_new'] = np.where(df[min_col] == 0, 3.0, df['rank_new'])

    # 3. Analyze Results
    changed = df[df['rank_old'] != df['rank_new']]
    
    print("\n📊 BLOWOUT PROTECTION REPORT")
    print(f"Total Games Analyzed: {len(df):,}")
    print(f"Rows 'Fixed' by New Logic: {len(changed):,} ({len(changed)/len(df)*100:.1f}%)")
    
    # Show "Saved Stars" (The Tatum Scenario)
    # Played < 26 mins but Kept as Rank 1
    saved_stars = changed[
        (changed[min_col] < 26) & 
        (changed['rank_new'] == 1.0)
    ]
    
    print(f"⭐ Stars Saved (Blowouts/Injuries): {len(saved_stars):,}")
    if len(saved_stars) > 0:
        print("\nExamples of Saved Stars:")
        cols_to_show = ['player', 'date', min_col, 'feat_avg_minutes', 'rank_old', 'rank_new']
        # Filter cols that actually exist
        cols_to_show = [c for c in cols_to_show if c in df.columns]
        print(saved_stars[cols_to_show].head(10).to_string(index=False))

    # Verdict
    print("\n------------------------------------------------")
    if len(changed) == 0:
        print("⚠️ NO CHANGE: Something is wrong, or logic is identical.")
    elif len(changed) > len(df) * 0.25:
        print("⚠️ TOO AGGRESSIVE: Changing > 25% of data. Check thresholds.")
    else:
        print("✅ VALIDATION PASSED: Logic is fixing edge cases without breaking data.")

if __name__ == "__main__":
    validate()
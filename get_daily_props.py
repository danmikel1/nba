import requests
import pandas as pd
import time

BDL_API_KEY = "c621f649-52f1-4219-8e77-f3d6e6624148"
HEADERS = {'Authorization': BDL_API_KEY}

def get_daily_props(target_date: str):
    print(f"--- Starting Prop Pull for {target_date} ---")
    
    # 1. Get all Game IDs for the day (Counts as 1 Request)
    games_url = f"https://api.balldontlie.io/v1/games?dates[]={target_date}"
    games_res = requests.get(games_url, headers=HEADERS)
    
    if games_res.status_code != 200:
        print("Failed to fetch games.")
        return
        
    games = games_res.json().get("data", [])
    if not games:
        print("No games found for this date.")
        return
        
    print(f"Found {len(games)} games. Commencing safe-pull to avoid Rate Limits...")
    
    all_daily_props = []
    
    # 2. Loop through each game safely
    for index, game in enumerate(games, 1):
        game_id = game["id"]
        matchup = f"{game['visitor_team']['abbreviation']} @ {game['home_team']['abbreviation']}"
        
        print(f"[{index}/{len(games)}] Pulling props for {matchup} (Game ID: {game_id})...")
        
        props_url = "https://api.balldontlie.io/v2/odds/player_props"
        props_res = requests.get(props_url, headers=HEADERS, params={'game_id': game_id})
        
        if props_res.status_code == 200:
            prop_data = props_res.json().get("data", [])
            
            # Append game context to the prop data
            for prop in prop_data:
                prop["matchup"] = matchup
                prop["date"] = target_date
                
            all_daily_props.extend(prop_data)
        else:
            print(f"  -> Error pulling {matchup}: Status {props_res.status_code}")

        # --- THE CRITICAL STEP ---
        # Free tier allows 5 requests per minute (1 request every 12 seconds).
        # We sleep for 13 seconds to guarantee we never trigger the ban.
        if index < len(games):
            print("  -> Sleeping 13 seconds for Rate Limit compliance...")
            time.sleep(13) 

    # 3. Compile and Export
    if all_daily_props:
        df = pd.DataFrame(all_daily_props)
        filename = f"nba_props_{target_date}.csv"
        df.to_csv(filename, index=False)
        print(f"\nSUCCESS: Exported {len(df)} prop lines to {filename}!")
    else:
        print("\nFailed to gather any prop data.")

# Run it once for the day
get_daily_props("2026-03-26")
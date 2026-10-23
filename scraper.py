import requests
import pandas as pd
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def classify_prop_type(market_name):
    name_lower = market_name.lower()
    if 'points + rebounds + assists' in name_lower: return 'PRA'
    if 'points + rebounds' in name_lower: return 'PR'
    if 'points + assists' in name_lower: return 'PA'
    if 'rebounds + assists' in name_lower: return 'RA'
    if 'points' in name_lower and '3-point' not in name_lower: return 'PTS'
    if 'rebounds' in name_lower: return 'REB'
    if 'assists' in name_lower: return 'AST'
    return None

def fetch_specific_game_props(game_id, matchup):
    url = "https://ph1xbet.ph/LineFeed/GetGameZip"
    params = {
        "id": game_id,
        "lng": "en",
        "isSubGames": "true",
        "GroupEvents": "true",
        "allEventsGroupSubGames": "true"
    }
    
    # Fortified browser signature
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://ph1xbet.ph/en/line/basketball',
        'Connection': 'keep-alive',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin'
    }
    
    response = requests.get(url, params=params, headers=headers, verify=False)
    
    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError:
        print(f"Data Payload Failure. HTTP Status: {response.status_code}")
        print(f"Server returned non-JSON response. Preview:\n{response.text[:250]}")
        return pd.DataFrame()
        
    extracted_data = []
    subgames = payload.get('Value', {}).get('SG', [])
    
    for sg in subgames:
        raw_market_name = sg.get('TG', '') 
        prop_type = classify_prop_type(raw_market_name)
        
        if prop_type:
            name_parts = raw_market_name.split('.')
            player_name = name_parts[0].strip() if len(name_parts) > 1 else raw_market_name
            
            events = sg.get('E', [])
            for event in events:
                outcome_id = event.get('T')
                if outcome_id == 9:
                    side = "Over"
                elif outcome_id == 10:
                    side = "Under"
                else:
                    continue 
                
                extracted_data.append({
                    'Matchup': matchup,
                    'Player': player_name,
                    'Market': prop_type,
                    'Side': side,
                    'Line': event.get('P'),
                    'Odds': event.get('C')
                })
                
    return pd.DataFrame(extracted_data)

target_game_id = 317593346
matchup_name = "CHA vs NYK"

print(f"Executing extraction for Game ID: {target_game_id}...")
df = fetch_specific_game_props(target_game_id, matchup_name)

if not df.empty:
    df.to_csv(f'1xbet_props_{target_game_id}.csv', index=False)
    print(f"Extraction complete. {len(df)} lines saved.")
    print(df.head())
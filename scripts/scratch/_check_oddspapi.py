import json, os

# Check oddspapi_cache for player prop lines
sample = os.listdir('data/oddspapi_cache')[0]
with open(f'data/oddspapi_cache/{sample}', encoding='utf-8') as f:
    data = json.load(f)

print(type(data))
if isinstance(data, dict):
    print("Keys:", list(data.keys()))
    bm = data.get('bookmakers', {})
    if bm:
        print(f"Bookmaker count: {len(bm)}")
        # Could be dict or list
        if isinstance(bm, dict):
            first_key = list(bm.keys())[0]
            b = bm[first_key]
            print(f"First bookmaker: {first_key}")
        else:
            b = bm[0]
        if isinstance(b, dict):
            print("Bookmaker value keys:", list(b.keys())[:10])
            mkts = b.get('markets', b)
            if isinstance(mkts, dict):
                for mk, mv in list(mkts.items())[:3]:
                    print(f"  Market: {mk}")
                    if isinstance(mv, dict):
                        for ok, ov in list(mv.items())[:5]:
                            print(f"    {ok}: {ov}")
                    elif isinstance(mv, list):
                        for item in mv[:3]:
                            print(f"    {item}")
            elif isinstance(mkts, list):
                for m in mkts[:3]:
                    key = m.get("key", "?")
                    outcomes = m.get("outcomes", [])
                    print(f"  Market: {key}")
                    for o in outcomes[:3]:
                        print(f"    {o}")
        else:
            print(f"Bookmaker value type: {type(b)}")
            print(str(b)[:500])

import os

import requests

BASE_URL = "https://api.oddspapi.io/v4"
BOOKMAKER = "betway"
NBA_TOURNAMENT_ID = 132
SPORT_ID = 11

api_key = os.environ.get("ODDSPAPI_API_KEY")
if not api_key:
    raise RuntimeError("Set ODDSPAPI_API_KEY before running this script")


def get_betway_nba_odds() -> list[dict]:
    response = requests.get(
        f"{BASE_URL}/odds-by-tournaments",
        params={
            "apiKey": api_key,
            "tournamentIds": str(NBA_TOURNAMENT_ID),
            "bookmakers": BOOKMAKER,
            "verbosity": 3,
            "oddsFormat": "decimal",
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        return []
    return [item for item in data if item.get("sportId") == SPORT_ID]


def detect_player_prop_rows(fixture: dict) -> list[dict]:
    rows = []
    betway = (fixture.get("bookmakerOdds") or {}).get(BOOKMAKER) or {}
    markets = betway.get("markets") or {}

    for market_id, market in markets.items():
        outcomes = market.get("outcomes") or {}
        for outcome_id, outcome in outcomes.items():
            players = outcome.get("players") or {}
            for player_id, player_entry in players.items():
                player_name = player_entry.get("playerName") if isinstance(player_entry, dict) else None
                if str(player_id) == "0" and not player_name:
                    continue
                rows.append(
                    {
                        "fixtureId": fixture.get("fixtureId"),
                        "marketId": market_id,
                        "outcomeId": outcome_id,
                        "playerId": str(player_id),
                        "playerName": player_name,
                        "line": player_entry.get("bookmakerOutcomeId") if isinstance(player_entry, dict) else None,
                        "price": player_entry.get("price") if isinstance(player_entry, dict) else None,
                        "active": player_entry.get("active") if isinstance(player_entry, dict) else None,
                    }
                )
    return rows


def main() -> None:
    fixtures = get_betway_nba_odds()
    if not fixtures:
        print("No NBA fixtures with Betway odds found.")
        return

    print(f"NBA fixtures returned by odds-by-tournaments: {len(fixtures)}")

    total_markets = 0
    total_player_rows = 0
    for index, fixture in enumerate(fixtures, 1):
        fixture_id = fixture.get("fixtureId")
        matchup = f"{fixture.get('participant1Name')} vs {fixture.get('participant2Name')}"
        betway = (fixture.get("bookmakerOdds") or {}).get(BOOKMAKER) or {}
        market_count = len(betway.get("markets") or {})
        total_markets += market_count

        print(f"\n[{index}/{len(fixtures)}] {matchup}")
        print(f"  fixtureId={fixture_id} start={fixture.get('startTime')} markets={market_count}")

        player_rows = detect_player_prop_rows(fixture)
        total_player_rows += len(player_rows)

        if not player_rows:
            print("  player-prop-style rows: 0")
            continue

        print(f"  player-prop-style rows: {len(player_rows)}")
        for row in player_rows[:8]:
            pname = row["playerName"] or f"player_id:{row['playerId']}"
            print(
                f"    market={row['marketId']} outcome={row['outcomeId']} {pname} "
                f"price={row['price']} line={row['line']} active={row['active']}"
            )

    print(f"\nSummary: fixtures={len(fixtures)} markets={total_markets} player_prop_rows={total_player_rows}")
    if total_player_rows == 0:
        print("No player-prop rows found for Betway in this NBA snapshot.")


if __name__ == "__main__":
    main()

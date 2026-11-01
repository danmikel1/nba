"""

odds_engine.py — The Odds API Integration (Game-Level Odds)



Fetches and caches real NBA spreads and game totals from The Odds API.

Your starter plan supports game-level markets (h2h, spreads, totals)

but NOT player props.



This replaces synthetic spread/total calculations (from team Net Rating

and Pace) with real Vegas lines — a significant accuracy improvement

for the feat_spread and feat_game_total model features.



Usage:

    engine = OddsEngine()



    # Daily refresh (1 API request, fetches all upcoming NBA games)

    engine.fetch_today()



    # Lookup for a specific game

    spread, total = engine.get_spread_and_total("LAL", "BOS", "2026-03-02", is_home=True)

    # Returns: (-3.5, 224.5) or (None, None)

"""



import json

import time

import logging

from pathlib import Path

from datetime import datetime, timedelta, date

from typing import Optional, Dict, List, Any, Tuple



import requests
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass



logger = logging.getLogger(__name__)



DATA_DIR = Path(__file__).resolve().parent / "data"

ODDS_CACHE_DIR = DATA_DIR / "odds_cache"

ODDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)



# The Odds API config

BASE_URL = "https://api.the-odds-api.com/v4"

SPORT = "basketball_nba"



# Game-level markets available on the starter plan

GAME_MARKETS = "spreads,totals"



# NBA team abbreviation aliases for fuzzy matching

_TEAM_ALIASES = {

    "atlanta hawks": "ATL", "hawks": "ATL",

    "boston celtics": "BOS", "celtics": "BOS",

    "brooklyn nets": "BKN", "nets": "BKN",

    "charlotte hornets": "CHA", "hornets": "CHA",

    "chicago bulls": "CHI", "bulls": "CHI",

    "cleveland cavaliers": "CLE", "cavaliers": "CLE", "cavs": "CLE",

    "dallas mavericks": "DAL", "mavericks": "DAL", "mavs": "DAL",

    "denver nuggets": "DEN", "nuggets": "DEN",

    "detroit pistons": "DET", "pistons": "DET",

    "golden state warriors": "GSW", "warriors": "GSW",

    "houston rockets": "HOU", "rockets": "HOU",

    "indiana pacers": "IND", "pacers": "IND",

    "los angeles clippers": "LAC", "la clippers": "LAC", "clippers": "LAC",

    "los angeles lakers": "LAL", "la lakers": "LAL", "lakers": "LAL",

    "memphis grizzlies": "MEM", "grizzlies": "MEM",

    "miami heat": "MIA", "heat": "MIA",

    "milwaukee bucks": "MIL", "bucks": "MIL",

    "minnesota timberwolves": "MIN", "timberwolves": "MIN", "wolves": "MIN",

    "new orleans pelicans": "NOP", "pelicans": "NOP",

    "new york knicks": "NYK", "knicks": "NYK",

    "oklahoma city thunder": "OKC", "thunder": "OKC",

    "orlando magic": "ORL", "magic": "ORL",

    "philadelphia 76ers": "PHI", "76ers": "PHI", "sixers": "PHI",

    "phoenix suns": "PHX", "suns": "PHX",

    "portland trail blazers": "POR", "trail blazers": "POR", "blazers": "POR",

    "sacramento kings": "SAC", "kings": "SAC",

    "san antonio spurs": "SAS", "spurs": "SAS",

    "toronto raptors": "TOR", "raptors": "TOR",

    "utah jazz": "UTA", "jazz": "UTA",

    "washington wizards": "WAS", "wizards": "WAS",

}





def _team_to_abbrev(team_name: str) -> str:

    """Convert a full team name to NBA abbreviation."""

    if not team_name:

        return ""

    lower = team_name.strip().lower()

    # Direct match

    if lower in _TEAM_ALIASES:

        return _TEAM_ALIASES[lower]

    # Try partial match (e.g., "Los Angeles Lakers" -> "lakers")

    for alias, abbrev in _TEAM_ALIASES.items():

        if alias in lower or lower in alias:

            return abbrev

    # Last resort: return as-is uppercased (3 chars)

    return team_name.strip().upper()[:3]





class OddsEngine:

    """

    Fetches and caches NBA game-level odds (spreads + totals) from The Odds API.



    Architecture:

    - Disk cache: data/odds_cache/YYYY-MM-DD.json (one file per date)

    - In-memory index: (home_abbrev, away_abbrev, date) -> game odds data

    - Also indexed by: (team_abbrev, date) -> game odds data (either side)

    - Fetch once per day, cache forever

    - Graceful fallback: returns None if no data (caller uses synthetic)

    """



    def __init__(self, api_key: str = None):

        self.api_key = api_key or self._load_api_key()

        # Primary index: (home_abbrev, away_abbrev, date_str) -> odds_data

        self._index: Dict[tuple, Dict] = {}

        # Team index: (team_abbrev, date_str) -> odds_data (for quick lookup by either team)

        self._team_index: Dict[tuple, Dict] = {}

        self._loaded_dates: set = set()

        self._requests_used = 0



        # Load existing cache into memory

        self._load_all_cached()



    def _load_api_key(self) -> str:

        """Load API key from config or environment."""

        import os

        key = os.environ.get("ODDS_API_KEY", "")

        if not key:

            config_path = DATA_DIR.parent / "config" / "deployment_config.json"

            if config_path.exists():

                try:

                    cfg = json.loads(config_path.read_text(encoding="utf-8"))

                    key = cfg.get("odds_api_key", "")

                except Exception:

                    pass

        return key



    @property

    def is_configured(self) -> bool:

        """Check if API key is set."""

        return bool(self.api_key)



    @property

    def cache_stats(self) -> Dict[str, int]:

        """Return cache statistics."""

        return {

            "dates_cached": len(self._loaded_dates),

            "total_games": len(self._index),

            "requests_this_session": self._requests_used,

        }



    # =========================================================================

    # CACHE MANAGEMENT

    # =========================================================================



    def _cache_path_for_date(self, date_str: str) -> Path:

        return ODDS_CACHE_DIR / f"{date_str}.json"



    def _load_all_cached(self):

        """Load all cached date files into memory index."""

        count = 0

        for f in ODDS_CACHE_DIR.glob("*.json"):

            date_str = f.stem

            if date_str in self._loaded_dates:

                continue

            try:

                data = json.loads(f.read_text(encoding="utf-8"))

                self._index_date(date_str, data)

                count += 1

            except Exception as e:

                logger.warning(f"Failed to load odds cache {f.name}: {e}")



        if count > 0:

            logger.info(f"Loaded odds cache: {count} dates, {len(self._index)} games")



    def _index_date(self, date_str: str, data: Dict):

        """Index all game odds from a single date's data."""

        self._loaded_dates.add(date_str)



        for game in data.get("games", []):

            home = game.get("home_abbrev", "")

            away = game.get("away_abbrev", "")

            if home and away:

                key = (home, away, date_str)

                self._index[key] = game

                # Also index by each team for easy lookup

                self._team_index[(home, date_str)] = game

                self._team_index[(away, date_str)] = game



    def _save_date_cache(self, date_str: str, data: Dict):

        path = self._cache_path_for_date(date_str)

        try:

            path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        except Exception as e:

            logger.warning(f"Failed to save odds cache for {date_str}: {e}")



    # =========================================================================

    # API FETCHING

    # =========================================================================



    def _fetch_oddspapi_bcgame(self) -> Optional[Dict]:

        """Fetch live NBA odds directly from BC.Game via OddsPapi.io."""

        import os

        oddspapi_key = os.environ.get("ODDSPAPI_API_KEY", "")

        if not oddspapi_key:

            return None

        try:

            from scripts.harvest_closing_odds import harvest_from_oddspapi_bcgame

            bc_games = harvest_from_oddspapi_bcgame(oddspapi_key)

            if not bc_games:

                return None



            games = []

            for bg in bc_games:

                home_abbrev = bg.get("home_team", "")

                away_abbrev = bg.get("away_team", "")

                if not home_abbrev or not away_abbrev:

                    continue

                games.append({

                    "home_team": home_abbrev,

                    "away_team": away_abbrev,

                    "home_abbrev": home_abbrev,

                    "away_abbrev": away_abbrev,

                    "game_date": bg.get("date", ""),

                    "spread": bg.get("spread_home"),

                    "total": bg.get("total"),

                    "spread_book": "BC.Game",

                    "total_book": "BC.Game",

                    "event_id": bg.get("event_id", ""),

                    "alternate_totals": bg.get("alternate_totals", []),

                    "alternate_spreads": bg.get("alternate_spreads", []),

                })



            return {

                "fetched_at": datetime.utcnow().isoformat(),

                "game_count": len(games),

                "games": games,

            }

        except Exception as e:

            logger.warning(f"BC.Game OddsPapi live fetch failed: {e}")

            return None



    def _fetch_live_odds(self) -> Optional[Dict]:

        """

        Fetch current NBA game odds (spreads + totals) for all upcoming games.

        Prioritizes BC.Game via OddsPapi.io, falling back to The Odds API.

        """

        # 1. Primary: BC.Game via OddsPapi

        bc_data = self._fetch_oddspapi_bcgame()

        if bc_data and bc_data.get("games"):

            logger.info("Using BC.Game odds (" + str(len(bc_data["games"])) + " games) via OddsPapi")

            return bc_data



        # 2. Fallback: The Odds API

        if not self.is_configured:

            logger.warning("Odds API key not configured — skipping fetch")

            return None



        url = f"{BASE_URL}/sports/{SPORT}/odds"

        params = {

            "apiKey": self.api_key,

            "regions": "us",

            "markets": GAME_MARKETS,

            "oddsFormat": "decimal",

        }



        try:

            logger.info("Fetching live NBA game odds (spreads + totals)...")

            resp = requests.get(url, params=params, timeout=30)

            self._requests_used += 1



            remaining = resp.headers.get("x-requests-remaining", "?")

            used = resp.headers.get("x-requests-used", "?")

            logger.info(f"  Odds API quota: {used} used, {remaining} remaining")



            if resp.status_code != 200:

                logger.error(f"Odds API error {resp.status_code}: {resp.text[:200]}")

                return None



            raw = resp.json()

            return self._parse_events(raw)



        except Exception as e:

            logger.error(f"Live odds fetch failed: {e}")

            return None



    def _parse_events(self, events: List[Dict]) -> Dict:

        """

        Parse The Odds API events into our normalized game odds format.



        Each event has:

        - home_team, away_team

        - bookmakers[] -> markets[] -> outcomes[]



        We extract the spread (home team perspective) and game total,

        preferring FanDuel > DraftKings > BetMGM > consensus.

        """

        games = []

        preferred = ["fanduel", "draftkings", "betmgm", "pointsbet", "caesars"]



        for event in events:

            home_team = event.get("home_team", "")

            away_team = event.get("away_team", "")

            commence = event.get("commence_time", "")

            event_id = event.get("id", "")



            home_abbrev = _team_to_abbrev(home_team)

            away_abbrev = _team_to_abbrev(away_team)



            # Extract date from commence_time (ISO format)

            game_date = ""

            if commence:

                try:

                    dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))

                    game_date = dt.strftime("%Y-%m-%d")

                except Exception:

                    game_date = datetime.utcnow().strftime("%Y-%m-%d")



            # Find best spread and total across bookmakers

            best_spread = None

            best_total = None

            best_spread_rank = 99

            best_total_rank = 99

            spread_book = "unknown"

            total_book = "unknown"



            for book in event.get("bookmakers", []):

                book_key = book.get("key", "unknown")

                book_rank = preferred.index(book_key) if book_key in preferred else 99



                for market in book.get("markets", []):

                    market_key = market.get("key", "")

                    outcomes = market.get("outcomes", [])



                    if market_key == "spreads" and book_rank < best_spread_rank:

                        # Find home team spread

                        for out in outcomes:

                            if out.get("name") == home_team:

                                best_spread = out.get("point")

                                best_spread_rank = book_rank

                                spread_book = book_key

                                break



                    elif market_key == "totals" and book_rank < best_total_rank:

                        # Find the over total

                        for out in outcomes:

                            if out.get("name", "").upper() == "OVER":

                                best_total = out.get("point")

                                best_total_rank = book_rank

                                total_book = book_key

                                break



            if home_abbrev and away_abbrev and (best_spread is not None or best_total is not None):

                games.append({

                    "home_team": home_team,

                    "away_team": away_team,

                    "home_abbrev": home_abbrev,

                    "away_abbrev": away_abbrev,

                    "game_date": game_date,

                    "spread": best_spread,  # Home team spread (negative = home favored)

                    "total": best_total,    # Over/under game total

                    "spread_book": spread_book,

                    "total_book": total_book,

                    "event_id": event_id,

                })



        result = {

            "fetched_at": datetime.utcnow().isoformat(),

            "game_count": len(games),

            "games": games,

        }



        logger.info(f"  Parsed {len(games)} games with spread/total odds")

        return result



    # =========================================================================

    # PUBLIC API

    # =========================================================================



    def get_game_odds(

        self,

        team1: str,

        team2: str = None,

        game_date: str = None,

    ) -> Optional[Dict]:

        """

        Look up game odds by team name(s) and date.



        Args:

            team1: Team abbreviation or full name (e.g., "LAL", "Lakers")

            team2: Optional second team (if omitted, finds any game for team1 on that date)

            game_date: Date string "YYYY-MM-DD"



        Returns:

            Dict with {"spread": float, "total": float, "home_team": str, ...}

            or None if not cached.

        """

        if not game_date:

            game_date = datetime.utcnow().strftime("%Y-%m-%d")



        abbrev1 = _team_to_abbrev(team1) if len(team1) > 3 else team1.upper()



        if team2:

            abbrev2 = _team_to_abbrev(team2) if len(team2) > 3 else team2.upper()

            # Try both home/away orderings

            result = self._index.get((abbrev1, abbrev2, game_date))

            if result:

                return result

            result = self._index.get((abbrev2, abbrev1, game_date))

            if result:

                return result

        else:

            # Single team lookup

            return self._team_index.get((abbrev1, game_date))



        return None



    def get_spread_and_total(

        self,

        player_team_abbrev: str,

        opp_abbrev: str,

        game_date: str,

        is_home: bool = True,

    ) -> Tuple[Optional[float], Optional[float]]:

        """

        Get spread (from player's perspective) and game total for a matchup.



        This is the main integration point for the backtest pipeline.



        Args:

            player_team_abbrev: Player's team abbreviation (e.g., "LAL")

            opp_abbrev: Opponent abbreviation (e.g., "BOS")

            game_date: Date string "YYYY-MM-DD"

            is_home: Whether the player's team is the home team



        Returns:

            (spread, total) tuple. Either or both can be None if not cached.

            Spread is from the player's team perspective (negative = favored).

        """

        if is_home:

            game = self._index.get((player_team_abbrev, opp_abbrev, game_date))

        else:

            game = self._index.get((opp_abbrev, player_team_abbrev, game_date))



        if not game:

            # Try the reverse

            if is_home:

                game = self._index.get((opp_abbrev, player_team_abbrev, game_date))

            else:

                game = self._index.get((player_team_abbrev, opp_abbrev, game_date))



        if not game:

            # Last resort: team index

            game = self._team_index.get((player_team_abbrev, game_date))



        if not game:

            return None, None



        spread = game.get("spread")  # Home team perspective

        total = game.get("total")



        # Adjust spread to player's team perspective

        if spread is not None:

            home_abbrev = game.get("home_abbrev", "")

            if player_team_abbrev != home_abbrev:

                spread = -spread  # Flip for away team



        return spread, total



    def fetch_today(self) -> int:

        """

        Fetch today's game odds and cache them.



        Returns: number of games fetched.

        Cost: 1 API request.

        """

        data = self._fetch_live_odds()

        if not data:

            return 0



        # Group games by date and save each date's cache

        games_by_date: Dict[str, list] = {}

        for game in data.get("games", []):

            gd = game.get("game_date", "")

            if gd:

                games_by_date.setdefault(gd, []).append(game)



        total = 0

        for date_str, games in games_by_date.items():

            date_data = {

                "fetched_at": data["fetched_at"],

                "game_count": len(games),

                "games": games,

            }

            self._save_date_cache(date_str, date_data)

            self._index_date(date_str, date_data)

            total += len(games)



        logger.info(f"Fetched odds for {total} games across {len(games_by_date)} dates")

        return total



    def get_coverage_report(self) -> Dict[str, Any]:

        """Report on cache coverage."""

        if not self._index:

            return {"total_games": 0, "dates": 0}



        dates = set()

        spreads = 0

        totals = 0



        for game in self._index.values():

            gd = game.get("game_date", "")

            if gd:

                dates.add(gd)

            if game.get("spread") is not None:

                spreads += 1

            if game.get("total") is not None:

                totals += 1



        date_list = sorted(dates)

        return {

            "total_games": len(self._index),

            "dates": len(dates),

            "date_range": f"{date_list[0]} to {date_list[-1]}" if date_list else "none",

            "games_with_spread": spreads,

            "games_with_total": totals,

        }





# Module-level singleton (lazy)

_odds_engine: Optional[OddsEngine] = None





def get_odds_engine() -> OddsEngine:

    """Get or create the shared OddsEngine singleton."""

    global _odds_engine

    if _odds_engine is None:

        _odds_engine = OddsEngine()

    return _odds_engine


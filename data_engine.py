import pandas as pd
import numpy as np
from pathlib import Path
from nba_api.stats.endpoints import leaguegamelog, commonplayerinfo, leaguedashteamstats
import os
import time
import logging
from datetime import datetime

# Set up logger
logger = logging.getLogger(__name__)

class DataEngine:
    """
    Centralized, high-performance caching layer for NBA source data.
    Uses Parquet for efficient storage and LeagueGameLog for bulk fetching.
    
    V21.1: Historical DvP Support
    - Caches defense stats per-season to disk (nba_season_stats_{season}.pkl)
    - get_historical_dvp_rank(team_id, season) returns correct historical ranks
    - Fixes "time travel" bug where 2026 ranks were used for 2023 games
    """

    # Seasons to pre-cache for historical lookups
    HISTORICAL_SEASONS = ['2022-23', '2023-24', '2024-25', '2025-26']

    def __init__(self, data_dir: Path, season: str = "2025-26", allow_fetch: bool = True):
        """
        Initialize the DataEngine.

        Args:
            data_dir: Path to the data directory.
            season: The target season (e.g., "2024-25").
            allow_fetch: if False, the engine will never hit the NBA API and
                         will rely exclusively on whatever is cached to disk.
                         This is useful for offline operation or when the
                         stats.nba.com service is unreliable.
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.season = season
        self.allow_fetch = allow_fetch  # new toggle

        # Columns to be optimized (Safe List)
        self.numeric_cols = [
            'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
            'FGA', 'FG3M', 'FTA', 'PF', 'MIN_FLOAT'
        ]

        # V21.1: Per-season defense stats cache: {'2023-24': df, '2024-25': df, ...}
        self.season_defense_cache: dict[str, pd.DataFrame] = {}

        # Position cache: {player_id: 'PG'} — avoids repeated API calls per session
        self._position_cache: dict[int, str] = {}

        # disk file for persistent position cache (shared with DataLoader)
        self.pos_file = self.data_dir / 'player_positions.json'
        # if the file exists load whatever positions are already recorded
        if self.pos_file.exists():
            try:
                import json
                with open(self.pos_file, 'r') as f:
                    data = json.load(f)
                # JSON keys are strings; convert to int where possible
                raw = data.get('positions', {})
                self._position_cache = {int(k): v for k, v in raw.items() if k.isdigit()}
                logger.info(f"Loaded {len(self._position_cache)} player positions from disk cache")
            except Exception as e:
                logger.warning(f"Failed to load position cache: {e}")

        # Load historical defense stats (from disk or API)
        self._load_historical_defense_stats()
        
        # Legacy compatibility: keep league_defense_df pointing to current season
        self.league_defense_df = self.season_defense_cache.get(self.season, pd.DataFrame())

    def _get_defense_stats_cache_path(self, season: str) -> Path:
        """Get the disk cache path for a season's defense stats."""
        return self.cache_dir / f"nba_season_stats_{season}.pkl"

    def _load_historical_defense_stats(self):
        """
        Load defense stats for all historical seasons.
        
        Priority:
        1. Load from disk cache (pkl files)
        2. If missing/stale, fetch from API and save to disk
        """
        for season in self.HISTORICAL_SEASONS:
            cache_path = self._get_defense_stats_cache_path(season)
            
            # Check if disk cache exists and is reasonably fresh (7 days for historical, 6 hours for current)
            ttl_hours = 6.0 if season == self.season else 168.0  # 7 days for historical
            
            if self._is_cache_fresh(cache_path, ttl_hours):
                try:
                    import joblib
                    df = joblib.load(cache_path)
                    self.season_defense_cache[season] = df
                    logger.debug(f"Loaded defense stats for {season} from disk cache")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load defense cache for {season}: {e}")
            
            # If API fetches are disabled, skip network call and keep whatever cache we have
            if not self.allow_fetch:
                if cache_path.exists():
                    logger.warning(f"API fetching disabled, using stale cache for {season}")
                    try:
                        import joblib
                        df = joblib.load(cache_path)
                        self.season_defense_cache[season] = df
                    except Exception:
                        self.season_defense_cache[season] = pd.DataFrame()
                else:
                    logger.warning(f"API fetching disabled and no cache for {season}")
                    self.season_defense_cache[season] = pd.DataFrame()
                continue

            # Fetch from API and cache
            df = self._fetch_season_defense_stats(season)
            if not df.empty:
                self.season_defense_cache[season] = df
                # Save to disk
                try:
                    import joblib
                    joblib.dump(df, cache_path)
                    logger.info(f"Cached defense stats for {season} to disk")
                except Exception as e:
                    logger.warning(f"Failed to save defense cache for {season}: {e}")
        
        loaded_count = len([s for s in self.season_defense_cache if not self.season_defense_cache[s].empty])
        logger.info(f"Loaded defense stats for {loaded_count}/{len(self.HISTORICAL_SEASONS)} seasons")

    def _fetch_season_defense_stats(self, season: str) -> pd.DataFrame:
        """
        Fetch defense stats for a specific season from NBA API.
        
        Args:
            season: Season string (e.g., '2023-24')
            
        Returns:
            pd.DataFrame: League defense stats with rankings for that season
        """
        if not self.allow_fetch:
            logger.warning(f"API fetching disabled; skipping defense stats fetch for {season}")
            return pd.DataFrame()

        try:
            # Rate limit
            time.sleep(0.6)
            
            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                per_mode_detailed='PerGame',
                measure_type_detailed_defense='Opponent'
            )
            df = team_stats.get_data_frames()[0]
            
            if df.empty:
                logger.warning(f"No league defense stats found for season {season}")
                return pd.DataFrame()
            
            logger.info(f"Fetched defense stats for {season}: {len(df)} teams")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch league defense stats for {season}: {e}")
            return pd.DataFrame()

    def _fetch_league_defense_stats(self) -> pd.DataFrame:
        """
        Fetch and cache league defense stats for DvP calculations.
        
        DEPRECATED: Use _load_historical_defense_stats() and get_historical_dvp_rank() instead.
        Kept for backwards compatibility.
        
        Returns:
            pd.DataFrame: League defense stats with rankings for current season.
        """
        # Return from cache if available
        if self.season in self.season_defense_cache:
            return self.season_defense_cache[self.season]
        
        # Fetch fresh
        df = self._fetch_season_defense_stats(self.season)
        if not df.empty:
            self.season_defense_cache[self.season] = df
        return df

    def _get_real_position(self, player_id: int) -> str:
        """
        Get the real position for a player using commonplayerinfo endpoint.

        - Checks in-memory cache first (avoids repeated API calls per session)
        - Retries up to 3 times with exponential backoff on timeout
        - Falls back to 'SF' if all attempts fail

        Returns normalized position code: PG, SG, SF, PF, or C.
        """
        # Cache hit — no API call needed
        if player_id in self._position_cache:
            logger.debug(f"Position cache hit for player {player_id}: {self._position_cache[player_id]}")
            return self._position_cache[player_id]

        # If we are not allowed to fetch, fall back immediately
        if not self.allow_fetch:
            logger.warning(f"API disabled; cannot fetch position for {player_id}, defaulting to SF")
            self._position_cache[player_id] = 'SF'
            self._write_position_cache()
            return 'SF'

        mapping = {
            'Guard': 'PG',
            'Guard-Forward': 'SG',
            'Forward-Guard': 'SF',
            'Forward': 'PF',
            'Forward-Center': 'PF',
            'Center-Forward': 'C',
            'Center': 'C'
        }

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                time.sleep(0.6)  # Respect NBA API rate limit
                player_info = commonplayerinfo.CommonPlayerInfo(
                    player_id=player_id,
                    timeout=15  # Shorter timeout — fail fast and retry
                )
                df = player_info.get_data_frames()[0]

                if df.empty:
                    logger.warning(f"No player info found for ID {player_id}")
                    break

                position_str = df.iloc[0].get('POSITION', '')
                normalized = mapping.get(position_str, 'SF')
                logger.debug(f"Player {player_id} position: {position_str} -> {normalized}")

                # Cache for the rest of the session
                self._position_cache[player_id] = normalized
                return normalized

            except Exception as e:
                wait = 2 ** attempt  # 2s, 4s, 8s
                if attempt < max_retries:
                    logger.warning(f"Position fetch attempt {attempt}/{max_retries} failed for player {player_id}: {e} — retrying in {wait}s")
                    time.sleep(wait)
                else:
                    logger.warning(f"Failed to get position for player {player_id} after {max_retries} attempts: {e}")

        # All retries exhausted — cache the fallback so we don't hammer the API again
        self._position_cache[player_id] = 'SF'
        self._write_position_cache()
        return 'SF'

    def _write_position_cache(self):
        """Persist the position map to disk (shares format with DataLoader)."""
        try:
            import json
            with open(self.pos_file, 'w') as f:
                json.dump({'time': datetime.now().isoformat(),
                           'positions': self._position_cache}, f)
        except Exception as e:
            logger.warning(f"Failed to write position cache: {e}")

    def _get_dvp_rank(self, opponent_id: int, position_code: str, season: str = None) -> int:
        """
        Get Defense vs. Position rank for an opponent team.
        
        V21.1: Now supports historical season lookup to fix time-travel bug.
        
        Args:
            opponent_id: Team ID of the opponent
            position_code: Player position (PG/SG/SF/PF/C)
            season: Optional season string for historical lookup (e.g., '2023-24')
        
        Returns:
            feat_opp_rank_vs_pos (1=Best Defense, 30=Worst Defense).
        """
        # Use historical lookup if season provided
        if season:
            return self.get_historical_dvp_rank(opponent_id, position_code, season)
        
        # Legacy: use current season
        if self.league_defense_df.empty:
            logger.warning("League defense stats not available")
            return 15  # Neutral rank
        
        return self._lookup_dvp_rank_in_df(self.league_defense_df, opponent_id, position_code)

    def get_historical_dvp_rank(self, opponent_id: int, position_code: str, season: str) -> int:
        """
        Get Defense vs. Position rank for a specific historical season.
        
        V21.1 FIX: This is the core fix for the "time travel" bug.
        When generating training data for a Nov 2023 game, this returns
        the team's defensive rank from the 2023-24 season, not current.
        
        Args:
            opponent_id: Team ID of the opponent
            position_code: Player position (PG/SG/SF/PF/C)
            season: Season string (e.g., '2023-24')
            
        Returns:
            int: DvP rank (1=Best Defense, 30=Worst Defense)
        """
        # Try exact season match
        if season in self.season_defense_cache:
            df = self.season_defense_cache[season]
            if not df.empty:
                return self._lookup_dvp_rank_in_df(df, opponent_id, position_code)
        
        # Fallback: find closest available season
        available_seasons = sorted([s for s in self.season_defense_cache.keys() 
                                    if not self.season_defense_cache[s].empty], reverse=True)
        
        if not available_seasons:
            logger.warning(f"No defense stats available for any season")
            return 15
        
        # Use most recent available
        fallback_season = available_seasons[0]
        logger.debug(f"Season {season} not in cache, falling back to {fallback_season}")
        df = self.season_defense_cache[fallback_season]
        return self._lookup_dvp_rank_in_df(df, opponent_id, position_code)

    def _lookup_dvp_rank_in_df(self, df: pd.DataFrame, opponent_id: int, position_code: str) -> int:
        """
        Internal helper to look up DvP rank in a defense stats DataFrame.
        
        Args:
            df: Defense stats DataFrame with TEAM_ID and rank columns
            opponent_id: Team ID to look up
            position_code: Position code (PG/SG/SF/PF/C)
            
        Returns:
            int: DvP rank (1-30), 15 if not found
        """
        # Find the opponent team
        opp_row = df[df['TEAM_ID'] == opponent_id]
        if opp_row.empty:
            logger.warning(f"Opponent team {opponent_id} not found in defense stats")
            return 15
        
        # Get rank based on position
        if position_code in ['PG', 'SG']:
            rank_col = 'OPP_AST_RANK'
        elif position_code in ['C', 'PF']:
            rank_col = 'OPP_REB_RANK'
        else:  # SF
            rank_col = 'OPP_PTS_RANK'
        
        if rank_col in opp_row.columns:
            rank = opp_row[rank_col].iloc[0]
            return int(rank)
        else:
            logger.warning(f"Rank column {rank_col} not found in defense stats")
            return 15

    def _get_cache_path(self, season: str) -> Path:
        """Get the path for the season's parquet cache file."""
        return self.cache_dir / f"league_games_{season}.parquet"

    def _is_cache_fresh(self, filepath: Path, ttl_hours: float = 6.0) -> bool:
        """Check if the cache file exists and is fresh."""
        if not filepath.exists():
            return False

        mtime = filepath.stat().st_mtime
        age_hours = (time.time() - mtime) / 3600.0
        return age_hours < ttl_hours

    def get_master_df(self, season: str = None, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get the master DataFrame for a season (LeagueGameLog).

        Logic:
        1. Check local Parquet cache.
        2. Validate freshness.
        3. Load if valid.
        4. Else, fetch from API, optimize, atomic write, return.

        Args:
            season: Target season (defaults to self.season).
            force_refresh: Whether to force an API call.

        Returns:
            pd.DataFrame: The master dataframe.
        """
        target_season = season if season else self.season
        cache_path = self._get_cache_path(target_season)

        # Path A: Cache Hit
        if not force_refresh and self._is_cache_fresh(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                # logger.info(f"Loaded cached data for {target_season}")
                return df
            except Exception as e:
                logger.warning(f"Failed to read cache for {target_season}, refreshing: {e}")

        # Path B: Cache Miss/Stale
        return self._fetch_and_cache(target_season, cache_path)

    def _fetch_and_cache(self, season: str, cache_path: Path) -> pd.DataFrame:
        """Fetch data from API, optimize, and save to cache."""
        logger.info(f"Fetching fresh data for {season}...")

        try:
            # Call NBA API (Player-centric mode)
            # 1 second delay to respect rate limits if called in loop
            time.sleep(0.6)

            log = leaguegamelog.LeagueGameLog(
                season=season,
                player_or_team_abbreviation='P'
            )
            df = log.get_data_frames()[0]

            if df.empty:
                logger.warning(f"No data returned for {season}")
                return pd.DataFrame()

            # Optimize Dtypes
            # 1. Convert GAME_DATE to datetime
            if 'GAME_DATE' in df.columns:
                df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

            # 2. Parse MIN column to MIN_FLOAT if needed
            # LeagueGameLog usually returns MIN as float or int, but sometimes string "34.5" or "34:30"
            # We add MIN_FLOAT for consistency with the rest of the app
            if 'MIN' in df.columns:
                def parse_minutes(min_val):
                    try:
                        if pd.isna(min_val):
                            return 0.0
                        if isinstance(min_val, (float, int)):
                            return float(min_val)
                        if isinstance(min_val, str):
                            if ':' in min_val:
                                parts = min_val.split(':')
                                return float(parts[0]) + float(parts[1]) / 60
                            return float(min_val)
                        return float(min_val)
                    except (ValueError, IndexError, TypeError):
                        return 0.0

                df['MIN_FLOAT'] = df['MIN'].apply(parse_minutes)

            # 3. Optimize numeric columns
            for col in self.numeric_cols:
                if col in df.columns:
                    # Coerce to numeric (errors='coerce' turns non-numeric to NaN)
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    # Downcast to save memory
                    df[col] = pd.to_numeric(df[col], downcast='integer') if 'FLOAT' not in col else pd.to_numeric(df[col], downcast='float')

            # Ensure TEAM_ID and PLAYER_ID are int
            for col in ['TEAM_ID', 'PLAYER_ID']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(int)

            # Atomic Write
            temp_path = cache_path.with_suffix('.tmp')
            df.to_parquet(temp_path, index=False)
            os.replace(temp_path, cache_path)

            logger.info(f"Cached {len(df)} rows for {season}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch/cache data for {season}: {e}")
            # If fetch fails, try to return stale cache if exists
            if cache_path.exists():
                logger.warning("Returning stale cache due to API failure")
                return pd.read_parquet(cache_path)
            raise

    def get_player_logs(self, player_id: int, season: str = None) -> pd.DataFrame:
        """
        Efficiently filter the cached master DataFrame for a specific player.

        Args:
            player_id: ID of the player.
            season: Target season (defaults to self.season).

        Returns:
            pd.DataFrame: Filtered game logs.
        """
        df = self.get_master_df(season)
        if df.empty:
            return pd.DataFrame()

        # Ensure player_id is int for comparison
        try:
            player_id = int(player_id)
        except (ValueError, TypeError):
            return pd.DataFrame()

        filtered_df = df[df['PLAYER_ID'] == player_id].copy()

        # Sort by date
        if 'GAME_DATE' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('GAME_DATE')

        return filtered_df.reset_index(drop=True)

    def fetch_player_data(self, player_id: int, opponent_id: int, market: str, line: float, season: str = None, player_position: str = None) -> dict:
        """
        Fetch player data and calculate position and DvP features.
        
        V21.1: Now passes season to get_historical_dvp_rank to fix time-travel bug.
        
        Args:
            player_id: NBA player ID
            opponent_id: Opponent team ID
            market: Stat market (PTS, REB, AST, PRA, etc.)
            line: The betting line
            season: Target season - CRITICAL for historical DvP lookup
            player_position: Optional pre-fetched/overridden position code (PG/SG/SF/PF/C)
            
        Returns:
            dict: Features dictionary with position and DvP
        """
        target_season = season if season else self.season
        
        # Use provided position if available to avoid extra API calls
        if player_position is None:
            player_position = self._get_real_position(player_id)
        else:
            logger.debug(f"Using preloaded position for player {player_id}: {player_position}")
        
        # V21.1 FIX: Pass season to DvP lookup for historical accuracy
        dvp_rank = self._get_dvp_rank(opponent_id, player_position, season=target_season)
        
        return {
            'player_position': player_position,
            'feat_opp_rank_vs_pos': dvp_rank
        }

    def force_refresh(self, season: str = None):
        """Manually trigger an API update for a season."""
        self.get_master_df(season, force_refresh=True)

    def refresh_historical_defense_stats(self, seasons: list[str] = None):
        """
        Force refresh defense stats for specified seasons.
        
        Args:
            seasons: List of season strings to refresh. Defaults to HISTORICAL_SEASONS.
        """
        target_seasons = seasons if seasons else self.HISTORICAL_SEASONS
        
        for season in target_seasons:
            logger.info(f"Force refreshing defense stats for {season}...")
            df = self._fetch_season_defense_stats(season)
            
            if not df.empty:
                self.season_defense_cache[season] = df
                
                # Save to disk
                cache_path = self._get_defense_stats_cache_path(season)
                try:
                    import joblib
                    joblib.dump(df, cache_path)
                    logger.info(f"Saved defense stats for {season} to {cache_path}")
                except Exception as e:
                    logger.warning(f"Failed to save defense cache for {season}: {e}")
        
        # Update legacy reference
        self.league_defense_df = self.season_defense_cache.get(self.season, pd.DataFrame())

    def get_cached_seasons(self) -> list[str]:
        """Return list of seasons with cached defense stats."""
        return [s for s in self.season_defense_cache.keys() 
                if not self.season_defense_cache[s].empty]
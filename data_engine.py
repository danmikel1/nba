import pandas as pd
import numpy as np
from pathlib import Path
from nba_api.stats.endpoints import leaguegamelog
import os
import time
import logging
import uuid
from datetime import datetime

# Set up logger
logger = logging.getLogger(__name__)

class DataEngine:
    """
    Centralized, high-performance caching layer for NBA source data.
    Uses Parquet for efficient storage and LeagueGameLog for bulk fetching.
    """

    def __init__(self, data_dir: Path, season: str = "2024-25"):
        """
        Initialize the DataEngine.

        Args:
            data_dir: Path to the data directory.
            season: The target season (e.g., "2024-25").
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.season = season

        # Columns to be optimized (Safe List)
        self.numeric_cols = [
            'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
            'FGA', 'FG3M', 'FTA', 'PF', 'MIN_FLOAT'
        ]

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

            # Atomic Write with unique temp file to prevent race conditions
            temp_path = cache_path.with_suffix(f'.{uuid.uuid4()}.tmp')
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

    def force_refresh(self, season: str = None):
        """Manually trigger an API update for a season."""
        self.get_master_df(season, force_refresh=True)

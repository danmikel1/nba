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
    """

    def __init__(self, data_dir: Path, season: str = "2025-26"):
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

        # Cache league defense stats for DvP calculations
        self.league_defense_df = self._fetch_league_defense_stats()

    def _fetch_league_defense_stats(self) -> pd.DataFrame:
        """
        Fetch and cache league defense stats for DvP calculations.
        
        Returns:
            pd.DataFrame: League defense stats with rankings.
        """
        try:
            # Fetch league defense stats
            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=self.season,
                per_mode_detailed='PerGame',
                measure_type_detailed_defense='Opponent'
            )
            df = team_stats.get_data_frames()[0]
            
            if df.empty:
                logger.warning(f"No league defense stats found for season {self.season}")
                return pd.DataFrame()
            
            logger.info(f"Cached league defense stats for {len(df)} teams with {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch league defense stats: {e}")
            return pd.DataFrame()

    def _get_real_position(self, player_id: int) -> str:
        """
        Get the real position for a player using commonplayerinfo endpoint.
        
        Returns normalized position code: PG, SG, SF, PF, or C.
        """
        try:
            # Fetch player info
            player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
            df = player_info.get_data_frames()[0]
            
            if df.empty:
                logger.warning(f"No player info found for ID {player_id}")
                return 'SF'  # Fallback
            
            position_str = df.iloc[0].get('POSITION', '')
            
            # Normalize using the mapping
            mapping = {
                'Guard': 'PG', 
                'Guard-Forward': 'SG', 
                'Forward-Guard': 'SF',
                'Forward': 'PF', 
                'Forward-Center': 'PF', 
                'Center-Forward': 'C',
                'Center': 'C'
            }
            
            normalized = mapping.get(position_str, 'SF')  # Default to SF if not found
            logger.debug(f"Player {player_id} position: {position_str} -> {normalized}")
            return normalized
            
        except Exception as e:
            logger.warning(f"Failed to get position for player {player_id}: {e}")
            return 'SF'  # Fallback

    def _get_dvp_rank(self, opponent_id: int, position_code: str) -> int:
        """
        Get Defense vs. Position rank for an opponent team using cached league stats.
        
        Returns feat_opp_rank_vs_pos (1=Best Defense, 30=Worst Defense).
        """
        if self.league_defense_df.empty:
            logger.warning("League defense stats not available")
            return 15  # Neutral rank
        
        # Find the opponent team
        opp_row = self.league_defense_df[self.league_defense_df['TEAM_ID'] == opponent_id]
        if opp_row.empty:
            logger.warning(f"Opponent team {opponent_id} not found in league defense stats")
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
            logger.warning(f"Rank column {rank_col} not found in league defense stats")
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
        
        Args:
            player_id: NBA player ID
            opponent_id: Opponent team ID
            market: Stat market (PTS, REB, AST, PRA, etc.)
            line: The betting line
            season: Target season
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
        
        # Get DvP rank
        dvp_rank = self._get_dvp_rank(opponent_id, player_position)
        
        return {
            'player_position': player_position,
            'feat_opp_rank_vs_pos': dvp_rank
        }

    def force_refresh(self, season: str = None):
        """Manually trigger an API update for a season."""
        self.get_master_df(season, force_refresh=True)

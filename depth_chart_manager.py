"""
Depth Chart Manager - ESPN Depth Chart Scraper

Scrapes ESPN for NBA team depth charts to enable "Next Man Up" feature engineering.
When a starter is OUT, backup players are dynamically promoted in feat_depth_rank.

V21.1: Initial implementation
"""

import json
import time
import logging
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class DepthChartManager:
    """
    Manages NBA team depth charts scraped from ESPN.
    
    Key Features:
    - Scrapes ESPN depth chart pages
    - Caches results to disk (24-hour TTL)
    - Fuzzy name matching for player lookups
    - Dynamic promotion when starters are injured
    
    Usage:
        manager = DepthChartManager()
        rank = manager.get_player_depth_rank("T.J. McConnell", "IND")
        # Returns 1 (Starter), 2 (Backup), 3 (Reserve), or 4 (Deep Bench)
    """
    
    # ESPN URL template - requires both abbreviation AND full team name slug
    ESPN_DEPTH_URL = "https://www.espn.com/nba/team/depth/_/name/{team_abbr}/{team_slug}"
    
    # Map standard NBA abbreviations to ESPN format (abbrev, full-slug)
    NBA_TO_ESPN = {
        'ATL': ('atl', 'atlanta-hawks'),
        'BOS': ('bos', 'boston-celtics'),
        'BKN': ('bkn', 'brooklyn-nets'),
        'CHA': ('cha', 'charlotte-hornets'),
        'CHI': ('chi', 'chicago-bulls'),
        'CLE': ('cle', 'cleveland-cavaliers'),
        'DAL': ('dal', 'dallas-mavericks'),
        'DEN': ('den', 'denver-nuggets'),
        'DET': ('det', 'detroit-pistons'),
        'GSW': ('gs', 'golden-state-warriors'),
        'HOU': ('hou', 'houston-rockets'),
        'IND': ('ind', 'indiana-pacers'),
        'LAC': ('lac', 'la-clippers'),
        'LAL': ('lal', 'los-angeles-lakers'),
        'MEM': ('mem', 'memphis-grizzlies'),
        'MIA': ('mia', 'miami-heat'),
        'MIL': ('mil', 'milwaukee-bucks'),
        'MIN': ('min', 'minnesota-timberwolves'),
        'NOP': ('no', 'new-orleans-pelicans'),
        'NYK': ('ny', 'new-york-knicks'),
        'OKC': ('okc', 'oklahoma-city-thunder'),
        'ORL': ('orl', 'orlando-magic'),
        'PHI': ('phi', 'philadelphia-76ers'),
        'PHX': ('phx', 'phoenix-suns'),
        'POR': ('por', 'portland-trail-blazers'),
        'SAC': ('sac', 'sacramento-kings'),
        'SAS': ('sa', 'san-antonio-spurs'),
        'TOR': ('tor', 'toronto-raptors'),
        'UTA': ('utah', 'utah-jazz'),
        'WAS': ('wsh', 'washington-wizards'),
    }
    
    # Aliases for common abbreviation variations
    ABBREV_ALIASES = {
        'GS': 'GSW', 'NO': 'NOP', 'NY': 'NYK', 'SA': 'SAS', 'WSH': 'WAS',
        'PHO': 'PHX', 'NOR': 'NOP', 'SAN': 'SAS', 'UTH': 'UTA'
    }
    
    # Standard positions in depth chart order
    POSITIONS = ['PG', 'SG', 'SF', 'PF', 'C']
    
    def __init__(self, cache_dir: Path = None, cache_ttl_hours: float = 24.0):
        """
        Initialize the Depth Chart Manager.
        
        Args:
            cache_dir: Directory for cache files (defaults to data/)
            cache_ttl_hours: Hours before cache expires (default 24)
        """
        self.cache_dir = cache_dir or Path("data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "depth_charts.json"
        self.cache_ttl_hours = cache_ttl_hours
        
        # In-memory cache: {team_abbrev: {position: [player1, player2, ...]}}
        self._cache: Dict[str, Dict[str, List[str]]] = {}
        self._cache_timestamp: Optional[datetime] = None
        
        # Load from disk if available
        self._load_cache()
    
    def _load_cache(self):
        """Load depth chart cache from disk."""
        if not self.cache_file.exists():
            return
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check timestamp
            timestamp_str = data.get('_timestamp')
            if timestamp_str:
                self._cache_timestamp = datetime.fromisoformat(timestamp_str)
                
                # Check if cache is still valid
                age_hours = (datetime.now() - self._cache_timestamp).total_seconds() / 3600
                if age_hours > self.cache_ttl_hours:
                    logger.info(f"Depth chart cache expired ({age_hours:.1f}h old)")
                    return
            
            # Load team data
            self._cache = {k: v for k, v in data.items() if not k.startswith('_')}
            logger.info(f"Loaded depth charts for {len(self._cache)} teams from cache")
            
        except Exception as e:
            logger.warning(f"Failed to load depth chart cache: {e}")
    
    def _save_cache(self):
        """Save depth chart cache to disk."""
        try:
            data = dict(self._cache)
            data['_timestamp'] = datetime.now().isoformat()
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            self._cache_timestamp = datetime.now()
            logger.debug(f"Saved depth charts for {len(self._cache)} teams to cache")
            
        except Exception as e:
            logger.warning(f"Failed to save depth chart cache: {e}")
    
    def _is_cache_valid(self, team_abbrev: str) -> bool:
        """Check if cache for a team is valid."""
        if team_abbrev not in self._cache:
            return False
        if self._cache_timestamp is None:
            return False
        
        age_hours = (datetime.now() - self._cache_timestamp).total_seconds() / 3600
        return age_hours < self.cache_ttl_hours
    
    def _normalize_team_abbrev(self, team_abbrev: str) -> str:
        """Normalize team abbreviation to standard format."""
        abbrev = team_abbrev.upper()
        return self.ABBREV_ALIASES.get(abbrev, abbrev)
    
    def _get_espn_url(self, team_abbrev: str) -> str:
        """Get ESPN depth chart URL for a team."""
        abbrev = self._normalize_team_abbrev(team_abbrev)
        
        if abbrev in self.NBA_TO_ESPN:
            espn_abbr, team_slug = self.NBA_TO_ESPN[abbrev]
            return self.ESPN_DEPTH_URL.format(team_abbr=espn_abbr, team_slug=team_slug)
        
        # Fallback for unknown teams
        logger.warning(f"Unknown team abbreviation: {team_abbrev}")
        return self.ESPN_DEPTH_URL.format(team_abbr=team_abbrev.lower(), team_slug=team_abbrev.lower())
    
    def _scrape_team_depth_chart(self, team_abbrev: str) -> Dict[str, List[str]]:
        """
        Scrape depth chart for a team from ESPN.
        
        ESPN depth chart structure (2024-2025):
        - Table 0 has positions as rows: ['', 'PG', 'SG', 'SF', 'PF', 'C']
        - Table 1 has depth levels as columns and players as rows:
          Row 0: ['Starter', '2nd', '3rd', '4th', '5th']
          Row 1: PG players in depth order
          Row 2: SG players in depth order
          etc.
        
        Args:
            team_abbrev: Team abbreviation (e.g., 'IND', 'LAL')
            
        Returns:
            Dict mapping position to list of players in depth order
            e.g., {'PG': ['Tyrese Haliburton', 'T.J. McConnell', ...], ...}
        """
        url = self._get_espn_url(team_abbrev)
        
        try:
            # Rate limit
            time.sleep(0.5)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            depth_chart: Dict[str, List[str]] = {}
            
            # Find all tables
            tables = soup.find_all('table')
            
            if len(tables) >= 2:
                # ESPN structure: Table 0 = positions, Table 1 = players
                position_table = tables[0]
                players_table = tables[1]
                
                # Extract positions from Table 0 (skip header row)
                position_rows = position_table.find_all('tr')
                positions = []
                for row in position_rows:
                    cell_text = row.get_text(strip=True).upper()
                    if cell_text in self.POSITIONS:
                        positions.append(cell_text)
                
                # Extract players from Table 1
                player_rows = players_table.find_all('tr')
                
                # Skip header row (Starter, 2nd, 3rd, etc.)
                data_rows = []
                for row in player_rows:
                    cells = row.find_all(['td', 'th'])
                    # Use space separator so inline injury tags (span) are separated from name
                    cell_texts = [c.get_text(' ', strip=True) for c in cells]
                    # Skip if this is the header row
                    if cell_texts and cell_texts[0] in ['Starter', '1st', '']:
                        continue
                    data_rows.append(cell_texts)
                
                # Map positions to player rows
                for i, position in enumerate(positions):
                    if i < len(data_rows):
                        players = []
                        for name in data_rows[i]:
                            if name:
                                # Clean injury tags (O, DD, Q, D, P at the end)
                                # Only strip when preceded by a space to avoid corrupting
                                # names that naturally end with those letters (e.g. "Edgecombe")
                                clean_name = name.strip()
                                for tag in ['GTD', 'DD', 'O', 'Q', 'D', 'P']:
                                    if clean_name.endswith(' ' + tag):
                                        clean_name = clean_name[:-(len(tag) + 1)].strip()
                                        break
                                if clean_name and len(clean_name) > 2:
                                    players.append(clean_name)
                        depth_chart[position] = players
            
            # Fallback: Try text-based parsing if table parsing failed
            if sum(len(v) for v in depth_chart.values()) < 5:
                fallback = self._scrape_from_text(soup)
                if sum(len(v) for v in fallback.values()) > sum(len(v) for v in depth_chart.values()):
                    depth_chart = fallback
            
            # Remove empty positions
            depth_chart = {k: v for k, v in depth_chart.items() if v}
            
            if depth_chart:
                logger.info(f"Scraped depth chart for {team_abbrev}: {sum(len(v) for v in depth_chart.values())} players")
            else:
                logger.warning(f"No depth chart data found for {team_abbrev}")
            
            return depth_chart
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch depth chart for {team_abbrev}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error parsing depth chart for {team_abbrev}: {e}")
            return {}
    
    def _scrape_from_text(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """
        Parse depth chart from page text when table parsing fails.
        
        ESPN pages often have text like:
        "PG SG SF PF C Starter 2nd 3rd... Andrew Nembhard T.J. McConnell ..."
        
        We parse this by looking for the pattern after "Depth Chart" text.
        """
        depth_chart: Dict[str, List[str]] = {pos: [] for pos in self.POSITIONS}
        
        try:
            # Get full page text
            text = soup.get_text()
            
            # Find the depth chart section
            # Look for position headers followed by player names
            import re
            
            # Pattern: Find section between "PG SG SF PF C" and "Glossary"
            pattern = r'PG\s+SG\s+SF\s+PF\s+C\s+(.*?)Glossary'
            match = re.search(pattern, text, re.DOTALL)
            
            if not match:
                return {}
            
            content = match.group(1)
            
            # Remove known labels
            for label in ['Starter', '2nd', '3rd', '4th', '5th']:
                content = content.replace(label, ' ')
            
            # Split into names (capitalized words)
            # Names are typically "First Last" with possible middle initial
            name_pattern = r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
            names = re.findall(name_pattern, content)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_names = []
            for name in names:
                name_clean = name.strip()
                if name_clean and name_clean not in seen and len(name_clean) > 3:
                    seen.add(name_clean)
                    unique_names.append(name_clean)
            
            # ESPN lists in column order: all PGs, then all SGs, etc.
            # Each position has up to 5 players (Starter through 5th)
            # Total ~25 names, 5 per position
            
            # Distribute names to positions (5 per position, in order)
            max_depth = 5
            pos_idx = 0
            depth_idx = 0
            
            for name in unique_names:
                if pos_idx >= len(self.POSITIONS):
                    break
                    
                position = self.POSITIONS[pos_idx]
                depth_chart[position].append(name)
                depth_idx += 1
                
                if depth_idx >= max_depth:
                    depth_idx = 0
                    pos_idx += 1
            
            return depth_chart
            
        except Exception as e:
            logger.warning(f"Text-based parsing failed: {e}")
            return {}
    
    def _scrape_fallback(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Fallback scraper - delegates to text-based parsing."""
        return self._scrape_from_text(soup)
    
    def get_team_depth_chart(self, team_abbrev: str, force_refresh: bool = False) -> Dict[str, List[str]]:
        """
        Get depth chart for a team.
        
        Args:
            team_abbrev: Team abbreviation
            force_refresh: Force re-scrape even if cached
            
        Returns:
            Dict mapping position to list of players
        """
        team_abbrev = team_abbrev.upper()
        
        # Check cache
        if not force_refresh and self._is_cache_valid(team_abbrev):
            return self._cache.get(team_abbrev, {})
        
        # Scrape fresh data
        depth_chart = self._scrape_team_depth_chart(team_abbrev)
        
        if depth_chart:
            self._cache[team_abbrev] = depth_chart
            self._save_cache()
        
        return depth_chart
    
    def _normalize_name(self, name: str) -> str:
        """Normalize player name for matching."""
        if not name:
            return ""
        # Lowercase, remove periods, extra spaces
        normalized = name.lower().replace('.', '').replace("'", "").strip()

        # Strip status tags a la "SUSP" / "OUT" that can be appended in depth chart names
        # or from scraped sources, e.g., "Luka Doncic SUSP".
        for status in ['susp', 'out', 'inj', 'questionable', 'doubtful']:
            if normalized.endswith(f' {status}'):
                normalized = normalized[: -len(status) - 1].strip()

        # Normalize some common name variations
        normalized = normalized.replace('nicolas', 'nic').replace('nicholas', 'nick')

        return ' '.join(normalized.split())  # Collapse whitespace
    
    def _fuzzy_match(self, target: str, candidates: List[str]) -> Optional[Tuple[str, int]]:
        """
        Find best fuzzy match for a player name.
        
        Args:
            target: Player name to find
            candidates: List of candidate names
            
        Returns:
            Tuple of (matched_name, index) or None
        """
        target_norm = self._normalize_name(target)
        target_parts = set(target_norm.split())
        
        best_match = None
        best_score = 0
        
        for idx, candidate in enumerate(candidates):
            cand_norm = self._normalize_name(candidate)
            cand_parts = set(cand_norm.split())
            
            # Exact match
            if target_norm == cand_norm:
                return (candidate, idx)
            
            # Check if one contains the other
            if target_norm in cand_norm or cand_norm in target_norm:
                score = len(min(target_norm, cand_norm, key=len))
                if score > best_score:
                    best_score = score
                    best_match = (candidate, idx)
                continue
            
            # Check last name match + first initial
            if len(target_parts) >= 2 and len(cand_parts) >= 2:
                target_last = max(target_parts, key=len)
                cand_last = max(cand_parts, key=len)
                
                if target_last == cand_last:
                    # Last names match - check first name/initial
                    target_first = min(target_parts, key=len)
                    cand_first = min(cand_parts, key=len)
                    
                    if target_first[0] == cand_first[0]:  # Same first initial
                        score = len(target_last) + 5
                        if score > best_score:
                            best_score = score
                            best_match = (candidate, idx)
            
            # Partial word overlap
            overlap = len(target_parts & cand_parts)
            if overlap > 0:
                score = overlap * 3
                if score > best_score:
                    best_score = score
                    best_match = (candidate, idx)
        
        # Require minimum score to return a match
        if best_score >= 3:
            return best_match
        
        return None
    
    def get_player_depth_rank(self, player_name: str, team_abbrev: str) -> int:
        """
        Get a player's depth chart rank.
        
        Args:
            player_name: Player's name
            team_abbrev: Team abbreviation
            
        Returns:
            1 = Starter
            2 = Backup
            3 = Reserve
            4 = Deep Bench / Not Found
        """
        if not player_name or not team_abbrev:
            return 4
        
        depth_chart = self.get_team_depth_chart(team_abbrev)
        
        if not depth_chart:
            logger.debug(f"No depth chart for {team_abbrev}")
            return 4
        
        # Find BEST rank across all positions (player may appear at multiple positions)
        best_rank = 4
        
        for position, players in depth_chart.items():
            # Try exact match first
            for idx, chart_player in enumerate(players):
                if self._normalize_name(player_name) == self._normalize_name(chart_player):
                    rank = min(idx + 1, 4)
                    best_rank = min(best_rank, rank)
                    break  # Found in this position, check others
            else:
                # Try fuzzy match if no exact match
                match = self._fuzzy_match(player_name, players)
                if match:
                    _, idx = match
                    rank = min(idx + 1, 4)
                    best_rank = min(best_rank, rank)
        
        if best_rank == 4:
            logger.debug(f"Player {player_name} not found in {team_abbrev} depth chart")
        
        return best_rank
    
    def get_starter_at_position(self, team_abbrev: str, position: str) -> Optional[str]:
        """
        Get the starter at a specific position.
        
        Args:
            team_abbrev: Team abbreviation
            position: Position code (PG, SG, SF, PF, C)
            
        Returns:
            Starter's name or None
        """
        depth_chart = self.get_team_depth_chart(team_abbrev)
        
        if not depth_chart:
            return None
        
        position = position.upper()
        if position in depth_chart and depth_chart[position]:
            return depth_chart[position][0]
        
        return None
    
    def get_player_position_and_rank(self, player_name: str, team_abbrev: str) -> Tuple[str, int]:
        """
        Get a player's position and depth rank.
        
        Args:
            player_name: Player's name
            team_abbrev: Team abbreviation
            
        Returns:
            Tuple of (position, rank) e.g., ('PG', 2)
        """
        depth_chart = self.get_team_depth_chart(team_abbrev)
        
        if not depth_chart:
            return ('SF', 4)  # Default
        
        for position, players in depth_chart.items():
            # Try exact match
            for idx, chart_player in enumerate(players):
                if self._normalize_name(player_name) == self._normalize_name(chart_player):
                    return (position, min(idx + 1, 4))
            
            # Try fuzzy match
            match = self._fuzzy_match(player_name, players)
            if match:
                _, idx = match
                return (position, min(idx + 1, 4))
        
        return ('SF', 4)  # Not found
    
    def get_effective_depth_rank(
        self, 
        player_name: str, 
        team_abbrev: str,
        injured_players: List[str] = None
    ) -> int:
        """
        Get player's effective depth rank considering injuries.
        
        This is the "Dynamic Promotion" algorithm:
        - If a player is Rank 2, but the Rank 1 player is injured, they become Rank 1
        - If Rank 1 and 2 are both injured, Rank 3 becomes Rank 1
        - Checks ALL positions player appears at and returns best effective rank
        
        Args:
            player_name: Player's name
            team_abbrev: Team abbreviation
            injured_players: List of injured player names (OUT status)
            
        Returns:
            Effective depth rank (1-4)
        """
        if not injured_players:
            injured_players = []
        
        depth_chart = self.get_team_depth_chart(team_abbrev)
        
        if not depth_chart:
            return 4
        
        # Normalize injured player names for matching
        normalized_injured = [self._normalize_name(p) for p in injured_players]
        
        # Check ALL positions player appears at and find best effective rank
        best_effective_rank = 4
        
        for position, players in depth_chart.items():
            # Find player's index at this position
            player_idx = None
            for idx, chart_player in enumerate(players):
                if self._normalize_name(player_name) == self._normalize_name(chart_player):
                    player_idx = idx
                    break
            
            if player_idx is None:
                # Try fuzzy match
                match = self._fuzzy_match(player_name, players)
                if match:
                    _, player_idx = match
            
            if player_idx is not None:
                base_rank = min(player_idx + 1, 4)
                
                # Count how many players ahead at THIS position are injured
                players_ahead_injured = 0
                for idx in range(player_idx):  # Check players ahead in depth chart
                    ahead_player = players[idx]
                    ahead_normalized = self._normalize_name(ahead_player)
                    
                    # Check if this player is injured
                    if ahead_normalized in normalized_injured:
                        players_ahead_injured += 1
                    else:
                        # Also try fuzzy match for injuries
                        for inj in injured_players:
                            if self._fuzzy_match(inj, [ahead_player]):
                                players_ahead_injured += 1
                                break
                
                # Promote by number of injured players ahead
                effective_rank = max(1, base_rank - players_ahead_injured)
                best_effective_rank = min(best_effective_rank, effective_rank)
        
        base_rank = self.get_player_depth_rank(player_name, team_abbrev)
        if best_effective_rank != base_rank:
            msg = (f"Dynamic Promotion: {player_name} ({team_abbrev}) "
                   f"base Rank {base_rank} -> effective Rank {best_effective_rank} "
                   f"(injured: {injured_players})")
            logger.info(msg)
            print(msg)
        else:
            logger.debug(f"{player_name} effective rank: {best_effective_rank} (base: {base_rank})")
        
        return best_effective_rank
    
    def refresh_all_teams(self):
        """Force refresh depth charts for all 30 NBA teams."""
        all_teams = list(self.NBA_TO_ESPN_ABBREV.keys())[:30]  # Main 30 teams
        
        for team in all_teams:
            logger.info(f"Refreshing depth chart for {team}...")
            self.get_team_depth_chart(team, force_refresh=True)
            time.sleep(1)  # Be nice to ESPN
        
        logger.info(f"Refreshed depth charts for {len(all_teams)} teams")

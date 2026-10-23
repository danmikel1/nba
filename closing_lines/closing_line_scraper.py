"""
closing_line_scraper.py — 1WIN/BC Game/Bet88 Closing Line Scraper (Optional)

⚠️  DISCLAIMER:
  - Works only if the sportsbook's HTML structure is stable.
  - Many sites (including 1WIN) use JavaScript rendering; Selenium is needed.
  - Some sites actively block scrapers (rate limit, IP ban).
  - Use at your own risk and respect ToS.

IF this doesn't work for your book, fall back to:
  1. closing_line_cli.py (fast manual CLI)
  2. closing_line_helper.py (Streamlit batch upload)

Usage:
    python closing_line_scraper.py --book 1win --player "Luka Doncic" --market pts
    python closing_line_scraper.py --book 1win --batch bets.json --output lines.csv
"""

import argparse
import sys
import json
import re
import time
from pathlib import Path
from typing import Dict, Optional, List
import logging

# Try imports, skip if missing
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OneWinScraper:
    """
    1WIN scraper using Selenium (required for JS-rendered content).
    
    Note: 1WIN heavily obfuscates their site. This is a best-effort template.
    If it breaks, the site likely changed their HTML structure.
    """
    
    BASE_URL = "https://1win.com"
    
    def __init__(self, headless: bool = True):
        """Initialize Selenium driver."""
        if not HAS_SELENIUM:
            raise ImportError("Selenium required. Install: pip install selenium")
        
        options = ChromeOptions()
        if headless:
            options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
        
        try:
            self.driver = webdriver.Chrome(options=options)
        except Exception as e:
            logger.error(f"Chrome driver init failed: {e}. Download chromedriver from https://chromedriver.chromium.org/")
            raise
    
    def search_player_prop(self, player_name: str, market: str) -> Optional[Dict]:
        """
        Search for a player prop (pts, reb, ast, etc.) and return current line.
        
        Returns: {'player': ..., 'market': ..., 'line': X.X, 'odds': Y.YY, 'url': ...}
        """
        try:
            # Navigate to basketball/NBA section
            logger.info(f"Searching for {player_name} {market.upper()}...")
            self.driver.get(f"{self.BASE_URL}/en/live/basketball")
            
            # Wait for page load (adjust selector as needed)
            wait = WebDriverWait(self.driver, 10)
            search_box = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[placeholder*='Search']"))
            )
            
            # Type player name
            search_box.clear()
            search_box.send_keys(player_name)
            time.sleep(2)
            
            # Find prop market (pts, reb, ast)
            # This is highly fragile — adjust selectors based on actual 1WIN structure
            market_map = {'pts': 'points', 'reb': 'rebounds', 'ast': 'assists', 'pa': 'points+assists'}
            market_term = market_map.get(market.lower(), market.lower())
            
            # Look for market elem
            page_source = self.driver.page_source
            logger.debug(f"Page contains '{player_name}': {player_name.lower() in page_source.lower()}")
            
            # Try to extract line from page (very rough regex)
            lines = re.findall(r'(\d+\.\d+)\s*[-+]\s*(\d+\.\d+)', page_source)
            if lines:
                line, odds = float(lines[0][0]), float(lines[0][1])
                return {
                    'player': player_name,
                    'market': market.upper(),
                    'line': line,
                    'odds': odds,
                    'url': self.driver.current_url,
                    'timestamp': time.time(),
                }
            
            logger.warning(f"Could not parse line for {player_name} {market}")
            return None
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return None
    
    def close(self):
        """Close Selenium driver."""
        if hasattr(self, 'driver'):
            self.driver.quit()


class BetRazorAPI:
    """
    Fallback: Use BetRazor (free tier, aggregates multiple sportsbooks including Asian ones).
    More stable than scraping but requires API key (free tier available).
    
    Sign up: https://betrazor.com/api
    """
    
    BASE_URL = "https://api.betrazor.com"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        if not HAS_BS4:
            raise ImportError("requests + BeautifulSoup required. Install: pip install requests beautifulsoup4")
    
    def search_player_prop(self, player_name: str, market: str) -> Optional[Dict]:
        """
        Query BetRazor for player prop across sportsbooks.
        Note: Free tier may have limited data; paid tier has more coverage.
        """
        if not self.api_key:
            logger.warning("No BetRazor API key; skipping.")
            return None
        
        try:
            # Example: search for Luka Doncic points
            params = {
                'q': f"{player_name} {market}",
                'api_key': self.api_key,
            }
            resp = requests.get(f"{self.BASE_URL}/search", params=params, timeout=5)
            if resp.status_code != 200:
                logger.warning(f"BetRazor returned {resp.status_code}")
                return None
            
            data = resp.json()
            logger.info(f"BetRazor found: {data}")
            
            # Parse result (structure varies; check BetRazor docs)
            if 'results' in data and len(data['results']) > 0:
                result = data['results'][0]
                return {
                    'player': player_name,
                    'market': market.upper(),
                    'line': result.get('line'),
                    'odds': result.get('odds'),
                    'books': result.get('available_at', []),
                }
            
            return None
        
        except Exception as e:
            logger.error(f"BetRazor query failed: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Scrape closing lines from 1WIN or similar sportsbooks"
    )
    parser.add_argument('--book', default='1win', help='Sportsbook (1win, betrazor, etc.)')
    parser.add_argument('--player', help='Player name to search')
    parser.add_argument('--market', default='pts', help='Market (pts, reb, ast, pa)')
    parser.add_argument('--batch', help='JSON file with list of bets to look up')
    parser.add_argument('--output', help='Output CSV with results')
    parser.add_argument('--api-key', help='API key (for BetRazor or paid services)')
    args = parser.parse_args()
    
    if args.book == 'betrazor':
        logger.info("Using BetRazor API (requires API key)")
        searcher = BetRazorAPI(api_key=args.api_key)
        if args.player:
            result = searcher.search_player_prop(args.player, args.market)
            if result:
                print(json.dumps(result, indent=2))
            else:
                print("No result found.")
    
    elif args.book == '1win':
        if not HAS_SELENIUM:
            print("❌ Selenium required for 1WIN scraping. Install: pip install selenium")
            print("   Also download ChromeDriver: https://chromedriver.chromium.org/")
            sys.exit(1)
        
        logger.info("🕷️  Initializing 1WIN scraper (Selenium)...")
        scraper = OneWinScraper(headless=True)
        
        try:
            if args.player:
                result = scraper.search_player_prop(args.player, args.market)
                if result:
                    print(json.dumps(result, indent=2))
                else:
                    print("No result found.")
            
            elif args.batch:
                # Batch mode: load bets from JSON, look up each
                with open(args.batch, 'r') as f:
                    bets = json.load(f)
                
                results = []
                for i, bet in enumerate(bets):
                    if i > 0:
                        time.sleep(2)  # Rate limit
                    
                    result = scraper.search_player_prop(
                        bet.get('player', ''),
                        bet.get('market', 'pts')
                    )
                    if result:
                        results.append(result)
                        print(f"  [{i}] {result['player']}: {result['line']}")
                
                if args.output:
                    import pandas as pd
                    df = pd.DataFrame(results)
                    df.to_csv(args.output, index=False)
                    logger.info(f"✅ Saved {len(results)} results to {args.output}")
        
        finally:
            scraper.close()
    
    else:
        print(f"Unknown book: {args.book}")
        sys.exit(1)


if __name__ == "__main__":
    main()

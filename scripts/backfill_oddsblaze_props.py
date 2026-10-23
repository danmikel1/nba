"""
backfill_oddsblaze_props.py — OddsBlaze Historical Player Prop Lines Backfill

Fetches historical NBA player prop lines using the OddsBlaze Odds Rewind API.
Stores pre-match snapshots so we can replace synthetic EMA-based lines in
ml_training_data.csv with real sportsbook lines.

OddsBlaze Odds Rewind endpoint:
  GET https://rewind.odds.oddsblaze.com/
  Params: key, sportsbook, league, timestamp, market_contains, main, live, price

Rate limits (trial): 30 requests/minute, 24-hour key validity.

Strategy:
  1. Read unique game dates from ml_training_data.csv
  2. For each date, call Rewind at 3 pre-game timestamps (noon, 6pm, 9pm ET)
     to cover early + late game slates
  3. Store raw JSON snapshots in data/oddsblaze_cache/YYYY-MM-DD_HHZ.json
  4. Never re-fetch a date/time already cached
  5. Rate-limit to 28 req/min (safe margin under 30)

After backfill, run:  scripts/merge_oddsblaze_lines.py  (to be built)
to join real lines into ml_training_data.csv.

Usage:
    python scripts/backfill_oddsblaze_props.py --key YOUR_KEY
    python scripts/backfill_oddsblaze_props.py --key YOUR_KEY --sportsbook fanduel
    python scripts/backfill_oddsblaze_props.py --key YOUR_KEY --from 2024-01-01 --to 2024-06-01
    python scripts/backfill_oddsblaze_props.py --coverage   # show cached date coverage
"""

import json
import time
import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "oddsblaze_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

REWIND_URL = "https://rewind.odds.oddsblaze.com/"

# Pre-game snapshot times (UTC).  NBA tip-offs are typically:
#   Early slate:  ~12:00 ET = 17:00 UTC
#   Main slate:   ~19:00 ET = 00:00 UTC (next day)
#   Late slate:   ~22:00 ET = 03:00 UTC (next day)
# We take snapshots ~1h before the earliest tip in each window.
SNAPSHOT_HOURS_UTC = [16, 23]  # 11am ET + 6pm ET covers most pre-game lines

# Rate-limiting
MAX_REQ_PER_MIN = 28  # conservative (limit is 30)
MIN_INTERVAL = 60.0 / MAX_REQ_PER_MIN  # ~2.14s between requests

# OddsBlaze market names that map to our training data
PLAYER_MARKETS = (
    "Player Points,"
    "Player Rebounds,"
    "Player Assists,"
    "Player Points + Rebounds + Assists,"
    "Player Points + Rebounds,"
    "Player Points + Assists,"
    "Player Rebounds + Assists,"
    "Player Threes Made"
)


def cache_path(date_str: str, hour_utc: int, sportsbook: str) -> Path:
    """Return cache file path for a given date/hour/sportsbook."""
    return CACHE_DIR / f"{date_str}_{hour_utc:02d}Z_{sportsbook}.json"


def is_cached(date_str: str, hour_utc: int, sportsbook: str) -> bool:
    """Check if we already have a cached snapshot."""
    return cache_path(date_str, hour_utc, sportsbook).exists()


def fetch_rewind_snapshot(
    api_key: str,
    sportsbook: str,
    date_str: str,
    hour_utc: int,
    price_format: str = "american",
) -> dict | None:
    """
    Fetch a single Odds Rewind snapshot for NBA player props.

    Returns parsed JSON or None on failure.
    """
    ts = f"{date_str}T{hour_utc:02d}:00:00Z"

    params = {
        "key": api_key,
        "sportsbook": sportsbook,
        "league": "nba",
        "timestamp": ts,
        "market_contains": "Player",
        "main": "true",
        "live": "false",
        "price": price_format,
    }

    try:
        resp = requests.get(REWIND_URL, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            logger.warning("Rate limited — sleeping 60s")
            time.sleep(60)
            return None
        elif resp.status_code == 404:
            # No data for this timestamp (before OddsBlaze coverage)
            logger.debug("No data for %s @ %02d:00 UTC", date_str, hour_utc)
            return None
        else:
            logger.warning(
                "HTTP %d for %s @ %02d:00 UTC: %s",
                resp.status_code, date_str, hour_utc, resp.text[:200],
            )
            return None
    except requests.RequestException as e:
        logger.error("Request failed for %s @ %02d:00 UTC: %s", date_str, hour_utc, e)
        return None


def get_training_dates(from_date: str = None, to_date: str = None) -> list[str]:
    """Get sorted unique dates from training data, optionally filtered."""
    csv_path = DATA_DIR / "ml_training_data.csv"
    if not csv_path.exists():
        logger.error("Training data not found: %s", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path, usecols=["date"], low_memory=False)
    dates = sorted(df["date"].unique())

    if from_date:
        dates = [d for d in dates if d >= from_date]
    if to_date:
        dates = [d for d in dates if d <= to_date]

    return dates


def show_coverage(sportsbook: str):
    """Show which dates are cached and which are missing."""
    all_dates = get_training_dates()
    cached_dates = set()
    for f in CACHE_DIR.glob(f"*_{sportsbook}.json"):
        date_part = f.stem.split("_")[0]
        cached_dates.add(date_part)

    covered = [d for d in all_dates if d in cached_dates]
    missing = [d for d in all_dates if d not in cached_dates]

    print(f"Sportsbook: {sportsbook}")
    print(f"Training dates: {len(all_dates)}")
    print(f"Cached dates:   {len(covered)}  ({100*len(covered)/max(len(all_dates),1):.1f}%)")
    print(f"Missing dates:  {len(missing)}")

    if missing:
        print(f"\nMissing range: {missing[0]} to {missing[-1]}")
        # Estimate requests needed
        snapshots_needed = sum(
            1
            for d in missing
            for h in SNAPSHOT_HOURS_UTC
            if not is_cached(d, h, sportsbook)
        )
        print(f"Snapshots to fetch: {snapshots_needed}")
        est_minutes = snapshots_needed * MIN_INTERVAL / 60
        print(f"Estimated time: {est_minutes:.0f} min at {MAX_REQ_PER_MIN} req/min")


def run_backfill(
    api_key: str,
    sportsbook: str,
    from_date: str = None,
    to_date: str = None,
    max_requests: int = None,
):
    """Main backfill loop."""
    dates = get_training_dates(from_date, to_date)
    logger.info("Target dates: %d (%s to %s)", len(dates), dates[0], dates[-1])

    # Build work queue: (date, hour) pairs not yet cached
    queue = []
    for d in dates:
        for h in SNAPSHOT_HOURS_UTC:
            if not is_cached(d, h, sportsbook):
                queue.append((d, h))

    logger.info("Snapshots to fetch: %d (already cached: %d)",
                len(queue),
                len(dates) * len(SNAPSHOT_HOURS_UTC) - len(queue))

    if not queue:
        logger.info("Nothing to fetch — all dates cached!")
        return

    if max_requests:
        queue = queue[:max_requests]
        logger.info("Capped to %d requests (--max-requests)", max_requests)

    total = len(queue)
    success = 0
    empty = 0
    errors = 0
    last_request_time = 0.0

    for i, (date_str, hour_utc) in enumerate(queue, 1):
        # Rate limiting
        elapsed = time.time() - last_request_time
        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)

        logger.info("[%d/%d] Fetching %s @ %02d:00 UTC ...", i, total, date_str, hour_utc)
        last_request_time = time.time()

        data = fetch_rewind_snapshot(api_key, sportsbook, date_str, hour_utc)

        if data is None:
            errors += 1
            # Write empty marker so we don't retry dates with no coverage
            marker = cache_path(date_str, hour_utc, sportsbook)
            marker.write_text(json.dumps({"_empty": True, "date": date_str}), encoding="utf-8")
            continue

        events = data.get("events", [])
        prop_count = sum(
            1 for ev in events for o in ev.get("odds", [])
            if "player" in o
        )

        if prop_count == 0:
            empty += 1
            # Still cache the response (it had game odds but no props)
            out = cache_path(date_str, hour_utc, sportsbook)
            out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
            logger.info("  -> %d events, 0 player props (saved anyway)", len(events))
        else:
            success += 1
            out = cache_path(date_str, hour_utc, sportsbook)
            out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
            logger.info("  -> %d events, %d player props", len(events), prop_count)

    logger.info(
        "Done!  success=%d  empty=%d  errors=%d  total=%d",
        success, empty, errors, total,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Backfill NBA player prop lines from OddsBlaze Odds Rewind"
    )
    parser.add_argument("--key", type=str, help="OddsBlaze API key")
    parser.add_argument("--sportsbook", default="draftkings",
                        help="Sportsbook ID (default: draftkings)")
    parser.add_argument("--from", dest="from_date", type=str,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_date", type=str,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--max-requests", type=int,
                        help="Cap total requests (for testing)")
    parser.add_argument("--coverage", action="store_true",
                        help="Show cache coverage and exit")
    parser.add_argument("--price", default="american",
                        help="Price format (default: american)")
    args = parser.parse_args()

    if args.coverage:
        show_coverage(args.sportsbook)
        return

    if not args.key:
        import os
        args.key = os.environ.get("ODDSBLAZE_API_KEY", "")

    if not args.key:
        print("ERROR: --key required or set ODDSBLAZE_API_KEY env var")
        sys.exit(1)

    run_backfill(
        api_key=args.key,
        sportsbook=args.sportsbook,
        from_date=args.from_date,
        to_date=args.to_date,
        max_requests=args.max_requests,
    )


if __name__ == "__main__":
    main()

from odds_engine import OddsEngine
oe = OddsEngine(api_key="your_key")

# One-time: fetch all dates this season (~120 API requests)
oe.backfill("2025-10-22", "2026-03-03", max_requests=140)

# Check coverage
print(oe.cache_stats)
print(oe.get_coverage_report())
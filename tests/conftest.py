# Ensure tests can import the project package/module when running from the repo root
# Adds the repository root to sys.path so `from nba_prediction import ...` works reliably.
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

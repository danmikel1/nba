"""
server_sync.py — Pulls harvested NBA Game Odds & Player Props from the TrueNAS Home Server
via the shared Git repository (`nba_odds_harvester`) and mirrors them into
`C:/Users/DM/Desktop/nba/data/harvested_odds`.

Usage (CLI):
    python server_sync.py
"""

import logging
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

NBA_APP_DIR = Path(__file__).resolve().parent
LOCAL_HARVEST_DIR = NBA_APP_DIR / "data" / "harvested_odds"
DEFAULT_HARVESTER_REPO = NBA_APP_DIR.parent / "nba_odds_harvester"


def get_harvester_repo_dir() -> Path:
    """Resolve the path to the local clone of `nba_odds_harvester`."""
    env_dir = os.getenv("HARVESTER_REPO_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    return DEFAULT_HARVESTER_REPO


def _merge_settled_games(src: Path, dst: Path) -> None:
    try:
        import pandas as pd
        df_dst = pd.read_csv(dst) if dst.exists() else pd.DataFrame()
        df_src = pd.read_csv(src) if src.exists() else pd.DataFrame()
        if df_dst.empty and df_src.empty:
            return
        combined = pd.concat([df_dst, df_src], ignore_index=True)
        if "date" in combined.columns and "matchup" in combined.columns:
            combined.drop_duplicates(subset=["date", "matchup"], keep="last", inplace=True)
        combined.to_csv(dst, index=False)
        combined.to_csv(src, index=False)
    except Exception as e:
        logger.warning(f"Error merging settled_games.csv: {e}")


def _merge_settled_props(src: Path, dst: Path) -> None:
    try:
        import pandas as pd
        df_dst = pd.read_parquet(dst) if dst.exists() else pd.DataFrame()
        df_src = pd.read_parquet(src) if src.exists() else pd.DataFrame()
        if df_dst.empty and df_src.empty:
            return
        combined = pd.concat([df_dst, df_src], ignore_index=True)
        subset_cols = [c for c in ["date", "player", "market", "side"] if c in combined.columns]
        if subset_cols:
            combined.drop_duplicates(subset=subset_cols, keep="last", inplace=True)
        combined.to_parquet(dst, index=False)
        combined.to_parquet(src, index=False)
    except Exception as e:
        logger.warning(f"Error merging settled_player_props.parquet: {e}")


def sync_from_home_server(repo_dir: Path | None = None, target_dir: Path | None = None) -> Dict[str, Any]:
    """
    1. Runs `git pull --rebase --autostash` inside the local `nba_odds_harvester` repo
       (if a remote `origin` is configured).
    2. Copies any new or updated `.json`, `.csv`, or `.parquet` files from
       `nba_odds_harvester/data/harvested_odds` into `nba/data/harvested_odds`
       (intelligently merging settled databases without data loss).
    """
    repo = Path(repo_dir) if repo_dir else get_harvester_repo_dir()
    dest = Path(target_dir) if target_dir else LOCAL_HARVEST_DIR
    dest.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "ok": False,
        "git_pulled": False,
        "git_status": "not_checked",
        "files_updated": 0,
        "total_files": 0,
        "latest_file": None,
        "message": "",
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }

    if not repo.exists():
        result["message"] = f"Harvester repo folder not found at {repo}"
        return result

    # Step 1: Git pull if .git and remote 'origin' exist
    git_dir = repo / ".git"
    if git_dir.exists():
        try:
            remote_proc = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if remote_proc.returncode == 0 and remote_proc.stdout.strip():
                pull_proc = subprocess.run(
                    ["git", "pull", "--rebase", "--autostash"],
                    cwd=str(repo),
                    capture_output=True,
                    text=True,
                    timeout=45,
                )
                if pull_proc.returncode == 0:
                    result["git_pulled"] = True
                    result["git_status"] = "pulled"
                else:
                    result["git_status"] = f"pull_error: {pull_proc.stderr.strip()[:120]}"
            else:
                result["git_status"] = "no_remote_origin"
        except Exception as e:
            result["git_status"] = f"git_exception: {e}"

    # Step 2: Copy new or modified files from repo/data/harvested_odds -> dest
    src_harvest = repo / "data" / "harvested_odds"
    if not src_harvest.exists():
        result["message"] = f"No data/harvested_odds folder found in {repo}"
        return result

    updated = 0
    total = 0
    latest_mtime = 0.0
    latest_name = None

    for src_file in sorted(src_harvest.glob("*")):
        if not src_file.is_file() or src_file.suffix.lower() not in (".json", ".csv", ".parquet"):
            continue
        total += 1
        src_stat = src_file.stat()
        if src_stat.st_mtime > latest_mtime:
            latest_mtime = src_stat.st_mtime
            latest_name = src_file.name

        dst_file = dest / src_file.name

        # Smart merge for master settled databases
        if src_file.name == "settled_games.csv":
            _merge_settled_games(src_file, dst_file)
            updated += 1
            continue
        elif src_file.name == "settled_player_props.parquet":
            _merge_settled_props(src_file, dst_file)
            updated += 1
            continue

        needs_copy = False
        if not dst_file.exists():
            needs_copy = True
        else:
            dst_stat = dst_file.stat()
            if src_stat.st_size != dst_stat.st_size or int(src_stat.st_mtime) > int(dst_stat.st_mtime):
                needs_copy = True

        if needs_copy:
            shutil.copy2(src_file, dst_file)
            updated += 1

    result["ok"] = True
    result["files_updated"] = updated
    result["total_files"] = total
    result["latest_file"] = latest_name

    if result["git_pulled"]:
        result["message"] = f"Pulled from GitHub & synced ({updated} updated / {total} total files)."
    elif result["git_status"] == "no_remote_origin":
        result["message"] = f"Synced local harvester folder ({updated} updated / {total} total files). Add GitHub remote to sync with TrueNAS."
    else:
        result["message"] = f"Synced {updated} updated / {total} total files ({result['git_status']})."

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = sync_from_home_server()
    print(f"[{res['timestamp']}] {res['message']}")

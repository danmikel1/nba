"""
bet_slip_share.py — Generates a sleek, dark-mode shareable PNG card of tracked bets
(Game Lines & Player Props) and copies it directly to the Windows clipboard.
"""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Font Helpers (Windows Segoe UI / Consolas with graceful fallback)
# ---------------------------------------------------------------------------
FONT_DIR = Path(r"C:\Windows\Fonts")


def _load_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if mono:
        candidates = ["consolab.ttf" if bold else "consola.ttf", "courbd.ttf" if bold else "cour.ttf"]
    else:
        candidates = [
            "segoeuib.ttf" if bold else "segoeui.ttf",
            "arialbd.ttf" if bold else "arial.ttf",
        ]

    for name in candidates:
        fpath = FONT_DIR / name
        if fpath.exists():
            try:
                return ImageFont.truetype(str(fpath), size=size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Color Palette (Sleek Dark Sportsbook / Discord Card Theme)
# ---------------------------------------------------------------------------
BG_DARK = (13, 17, 23)          # #0d1117 Deep canvas
CARD_BG = (22, 27, 34)          # #161b22 Row container
CARD_BG_ALT = (26, 32, 42)      # #1a202a Alternate row
BORDER_SUBTLE = (48, 54, 61)    # #30363d Hairline border
HEADER_GRAD_TOP = (20, 30, 48)  # Header banner background
ACCENT_CYAN = (56, 189, 248)    # #38bdf8 Primary highlight
ACCENT_GOLD = (250, 204, 21)    # #facc15 Pending / Sweet Spot
ACCENT_GREEN = (34, 197, 94)    # #22c55e Win / Positive edge
ACCENT_RED = (248, 113, 113)    # #f87171 Loss
ACCENT_BLUE = (96, 165, 250)    # #60a5fa Push / Game badge
ACCENT_PURPLE = (192, 132, 252) # #c084fc Player prop badge
TEXT_PRIMARY = (240, 246, 252)  # #f0f6fc Crisp white
TEXT_SECONDARY = (139, 148, 158)# #8b949e Muted slate
TEXT_DARK = (10, 14, 20)


def _is_game_bet(bet: Dict[str, Any]) -> bool:
    mkt = str(bet.get("market", "")).upper()
    btype = str(bet.get("bet_type", "")).lower()
    return mkt in ("SPREAD", "TOTAL") or btype in ("spread", "total")


def _format_american_or_decimal(bet: Dict[str, Any]) -> str:
    odds_dec = bet.get("odds_decimal")
    odds_raw = bet.get("odds")
    if odds_dec is not None:
        try:
            return f"{float(odds_dec):.2f}"
        except (ValueError, TypeError):
            pass
    if odds_raw is not None:
        try:
            val = float(odds_raw)
            if 1.01 <= val <= 20.0:
                return f"{val:.2f}"
            return f"{int(val):+d}"
        except (ValueError, TypeError):
            pass
    return "1.91"


def _draw_pill(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: Any,
    bg_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int] = TEXT_PRIMARY,
    pad_x: int = 12,
    pad_y: int = 5,
    radius: int = 8,
    border_color: Optional[Tuple[int, int, int]] = None,
) -> int:
    """Draw a rounded pill badge and return its width."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    w = tw + pad_x * 2
    h = th + pad_y * 2
    draw.rounded_rectangle(
        [x, y, x + w, y + h],
        radius=radius,
        fill=bg_color,
        outline=border_color,
        width=1 if border_color else 0,
    )
    draw.text((x + pad_x, y + pad_y - 1), text, font=font, fill=text_color)
    return w


def render_tracked_bets_png(
    bets: List[Dict[str, Any]],
    title: str = "NBA QUANT BET SLIP",
    subtitle: Optional[str] = None,
    show_stakes: bool = True,
) -> bytes:
    """
    Render a list of tracked bet dictionaries into a high-resolution PNG byte stream.
    """
    if not bets:
        bets = []

    width = 1080
    pad_outer = 32
    header_h = 152
    col_header_h = 40
    row_h = 84
    row_gap = 10
    footer_h = 56

    num_rows = max(1, len(bets))
    total_h = (
        pad_outer
        + header_h
        + 18
        + col_header_h
        + num_rows * (row_h + row_gap)
        + footer_h
        + pad_outer
    )

    img = Image.new("RGB", (width, total_h), color=BG_DARK)
    draw = ImageDraw.Draw(img)

    # Load fonts
    f_title = _load_font(28, bold=True)
    f_sub = _load_font(15, bold=False)
    f_stat_lbl = _load_font(12, bold=True)
    f_stat_val = _load_font(20, bold=True, mono=True)
    f_col = _load_font(13, bold=True)
    f_player = _load_font(19, bold=True)
    f_meta = _load_font(13, bold=False)
    f_pick = _load_font(16, bold=True, mono=True)
    f_num = _load_font(16, bold=True, mono=True)
    f_badge = _load_font(13, bold=True)
    f_footer = _load_font(13, bold=False)

    # -----------------------------------------------------------------------
    # 1. Header Banner
    # -----------------------------------------------------------------------
    hx0, hy0 = pad_outer, pad_outer
    hx1, hy1 = width - pad_outer, pad_outer + header_h
    draw.rounded_rectangle(
        [hx0, hy0, hx1, hy1],
        radius=16,
        fill=HEADER_GRAD_TOP,
        outline=(45, 65, 95),
        width=2,
    )
    # Accent top bar stripe
    draw.rounded_rectangle(
        [hx0 + 2, hy0 + 2, hx1 - 2, hy0 + 8],
        radius=4,
        fill=ACCENT_CYAN,
    )

    # Left side of header: Title & Date
    draw.text((hx0 + 26, hy0 + 24), title, font=f_title, fill=TEXT_PRIMARY)
    now_str = datetime.now().strftime("%b %d, %Y • %I:%M %p")
    sub_text = subtitle or f"Generated {now_str}  |  Model-Driven NBA Edges"
    draw.text((hx0 + 26, hy0 + 62), sub_text, font=f_sub, fill=TEXT_SECONDARY)

    # Category counts pill row
    game_count = sum(1 for b in bets if _is_game_bet(b))
    prop_count = len(bets) - game_count
    px = hx0 + 26
    py = hy0 + 98
    px += _draw_pill(draw, px, py, f"TOTAL PICKS: {len(bets)}", f_badge, (30, 41, 59), ACCENT_CYAN, border_color=(56, 189, 248)) + 10
    if game_count > 0:
        px += _draw_pill(draw, px, py, f"GAME LINES: {game_count}", f_badge, (23, 37, 84), ACCENT_BLUE) + 10
    if prop_count > 0:
        px += _draw_pill(draw, px, py, f"PLAYER PROPS: {prop_count}", f_badge, (59, 7, 100), ACCENT_PURPLE) + 10

    # Right side of header: Summary Stat Boxes (Record, Pending, P/L or Avg Edge)
    wins = sum(1 for b in bets if b.get("result") == "Win")
    losses = sum(1 for b in bets if b.get("result") == "Loss")
    pushes = sum(1 for b in bets if b.get("result") == "Push")
    pending = sum(1 for b in bets if b.get("result", "Pending") == "Pending")

    net_pnl = 0.0
    total_staked = 0.0
    for b in bets:
        st_val = float(b.get("stake", 0) or 0)
        res = b.get("result", "Pending")
        total_staked += st_val
        if res == "Win":
            try:
                od = float(b.get("odds_decimal") or 1.91)
            except (ValueError, TypeError):
                od = 1.91
            net_pnl += st_val * (od - 1.0)
        elif res == "Loss":
            net_pnl -= st_val

    valid_evs = [float(b.get("ev", b.get("predicted_ev", 0)) or 0) for b in bets if (b.get("ev") is not None or b.get("predicted_ev") is not None)]
    avg_ev = sum(valid_evs) / max(1, len(valid_evs)) if valid_evs else 0.0

    stat_boxes = []
    if wins + losses > 0:
        wr = wins / max(1, wins + losses) * 100
        rec_str = f"{wins}W-{losses}L" + (f"-{pushes}P" if pushes else "")
        stat_boxes.append(("RECORD", f"{rec_str} ({wr:.0f}%)", ACCENT_GREEN if wr >= 52.4 else TEXT_PRIMARY))
    else:
        stat_boxes.append(("PENDING", f"{pending} Active", ACCENT_GOLD))

    if show_stakes and total_staked > 0:
        if wins + losses > 0:
            pnl_col = ACCENT_GREEN if net_pnl >= 0 else ACCENT_RED
            stat_boxes.append(("NET P/L", f"{net_pnl:+,.0f}", pnl_col))
        else:
            stat_boxes.append(("STAKED", f"{total_staked:,.0f}", ACCENT_CYAN))
    elif not show_stakes and len(stat_boxes) == 1 and avg_ev != 0.0:
        stat_boxes.append(("AVG EV", f"{avg_ev:+.1%}", ACCENT_GREEN if avg_ev > 0 else TEXT_PRIMARY))

    # Draw stat boxes aligned to right of header
    bx_right = hx1 - 24
    for lbl, val, col in reversed(stat_boxes):
        bw = 175
        bh = 86
        bx0 = bx_right - bw
        by0 = hy0 + 34
        draw.rounded_rectangle(
            [bx0, by0, bx_right, by0 + bh],
            radius=12,
            fill=(15, 23, 42),
            outline=BORDER_SUBTLE,
            width=1,
        )
        draw.text((bx0 + 16, by0 + 14), lbl, font=f_stat_lbl, fill=TEXT_SECONDARY)
        draw.text((bx0 + 16, by0 + 38), val, font=f_stat_val, fill=col)
        bx_right = bx0 - 12

    # -----------------------------------------------------------------------
    # 2. Column Headers
    # -----------------------------------------------------------------------
    cy = hy1 + 18
    col_x_type = pad_outer + 16
    col_x_player = pad_outer + 115
    col_x_pick = pad_outer + 385
    col_x_proj = pad_outer + 595
    col_x_stake = pad_outer + 745
    col_x_status = width - pad_outer - 115

    draw.text((col_x_type, cy + 10), "MARKET", font=f_col, fill=TEXT_SECONDARY)
    draw.text((col_x_player, cy + 10), "PLAYER / MATCHUP", font=f_col, fill=TEXT_SECONDARY)
    draw.text((col_x_pick, cy + 10), "PICK & ODDS", font=f_col, fill=TEXT_SECONDARY)
    draw.text((col_x_proj, cy + 10), "MODEL PROJ", font=f_col, fill=TEXT_SECONDARY)
    if show_stakes:
        draw.text((col_x_stake, cy + 10), "STAKE & EV", font=f_col, fill=TEXT_SECONDARY)
    else:
        draw.text((col_x_stake, cy + 10), "EDGE & EV", font=f_col, fill=TEXT_SECONDARY)
    draw.text((col_x_status, cy + 10), "STATUS", font=f_col, fill=TEXT_SECONDARY)

    # -----------------------------------------------------------------------
    # 3. Bet Rows
    # -----------------------------------------------------------------------
    ry = cy + col_header_h
    if not bets:
        draw.rounded_rectangle(
            [pad_outer, ry, width - pad_outer, ry + row_h],
            radius=12,
            fill=CARD_BG,
            outline=BORDER_SUBTLE,
            width=1,
        )
        draw.text((pad_outer + 30, ry + 28), "No bets match the current filter.", font=f_player, fill=TEXT_SECONDARY)
        ry += row_h + row_gap

    for idx, bet in enumerate(bets):
        r_bg = CARD_BG if idx % 2 == 0 else CARD_BG_ALT
        res = str(bet.get("result", "Pending"))

        # Left accent strip color by status
        if res == "Win":
            strip_col = ACCENT_GREEN
            row_border = (30, 85, 55)
        elif res == "Loss":
            strip_col = ACCENT_RED
            row_border = (90, 40, 45)
        elif res == "Push":
            strip_col = ACCENT_BLUE
            row_border = BORDER_SUBTLE
        else:
            strip_col = ACCENT_CYAN
            row_border = BORDER_SUBTLE

        rx0, ry0 = pad_outer, ry
        rx1, ry1 = width - pad_outer, ry + row_h
        draw.rounded_rectangle(
            [rx0, ry0, rx1, ry1],
            radius=12,
            fill=r_bg,
            outline=row_border,
            width=1,
        )
        # Left vertical status bar
        draw.rounded_rectangle(
            [rx0 + 2, ry0 + 6, rx0 + 8, ry1 - 6],
            radius=3,
            fill=strip_col,
        )

        # Column 1: Market Badge (GAME vs PROP market)
        is_game = _is_game_bet(bet)
        mkt_str = str(bet.get("market", "PROP")).upper()
        if is_game:
            _draw_pill(draw, col_x_type, ry0 + 24, mkt_str, f_badge, (23, 37, 84), ACCENT_BLUE, pad_x=12, pad_y=6)
        else:
            _draw_pill(draw, col_x_type, ry0 + 24, mkt_str, f_badge, (46, 16, 101), ACCENT_PURPLE, pad_x=12, pad_y=6)

        # Column 2: Player / Matchup + Opponent & Date
        player_name = str(bet.get("player", "Unknown"))
        if len(player_name) > 24:
            player_name = player_name[:22] + ".."
        draw.text((col_x_player, ry0 + 16), player_name, font=f_player, fill=TEXT_PRIMARY)

        opp = str(bet.get("opponent", "") or "").strip()
        b_date = str(bet.get("date", "") or "").strip()
        b_tag = str(bet.get("tag", "") or "").strip()
        meta_parts = []
        if opp and not is_game and "@" not in player_name:
            if "@" in opp or "vs" in opp.lower():
                meta_parts.append(opp.replace("@", " @ "))
            elif opp not in player_name:
                meta_parts.append(f"vs {opp}")
        if b_date:
            meta_parts.append(b_date)
        if b_tag and b_tag not in ("legacy", "None"):
            meta_parts.append(f"[{b_tag}]")
        meta_line = "  •  ".join(meta_parts) if meta_parts else "NBA Official Slate"
        draw.text((col_x_player, ry0 + 46), meta_line, font=f_meta, fill=TEXT_SECONDARY)

        # Column 3: Pick & Odds Pill
        side = str(bet.get("side", "")).strip().upper()
        line_val = bet.get("line", 0)
        # Avoid repeating the line if `side` already contains it (e.g. "DET +1.5" or "UNDER 229.5")
        has_embedded_number = any(ch.isdigit() for ch in side)
        if has_embedded_number:
            pick_str = side
        else:
            try:
                line_f = float(line_val)
                if is_game and mkt_str == "SPREAD" and side not in ("OVER", "UNDER"):
                    pick_str = f"{side} {line_f:+.1f}"
                else:
                    pick_str = f"{side} {line_f:.1f}"
            except (ValueError, TypeError):
                pick_str = f"{side} {line_val}"

        odds_txt = _format_american_or_decimal(bet)
        _draw_pill(
            draw,
            col_x_pick,
            ry0 + 16,
            pick_str,
            f_pick,
            (15, 23, 42),
            ACCENT_CYAN,
            pad_x=12,
            pad_y=5,
            border_color=(56, 189, 248),
        )
        draw.text((col_x_pick + 4, ry0 + 50), f"Odds: @{odds_txt}", font=f_meta, fill=TEXT_SECONDARY)

        # Column 4: Model Projection (2 rows: Proj pill on top, Vegas Line below)
        proj_val = bet.get("proj")
        if proj_val is None:
            proj_val = bet.get("predicted_mean", 0)
        try:
            proj_f = float(proj_val)
            proj_txt = f"{proj_f:.1f}"
        except (ValueError, TypeError):
            proj_f = 0.0
            proj_txt = "N/A"

        _draw_pill(
            draw,
            col_x_proj,
            ry0 + 16,
            proj_txt,
            f_num,
            (15, 23, 42),
            TEXT_PRIMARY,
            pad_x=12,
            pad_y=5,
            border_color=(45, 55, 72),
        )
        try:
            line_disp = f"Line: {abs(float(line_val)):.1f}" if mkt_str == "SPREAD" else f"Line: {float(line_val):.1f}"
        except (ValueError, TypeError):
            line_disp = f"Line: {line_val}"
        draw.text((col_x_proj + 4, ry0 + 50), line_disp, font=f_meta, fill=TEXT_SECONDARY)

        # Compute Edge & EV
        ev_val = float(bet.get("ev", bet.get("predicted_ev", 0)) or 0)
        try:
            if is_game and mkt_str == "SPREAD":
                edge_pts = abs(abs(proj_f) - abs(float(line_val)))
            else:
                edge_pts = abs(proj_f - float(line_val))
        except (ValueError, TypeError):
            edge_pts = 0.0

        # Column 5: EDGE & EV (or STAKE & EV) — 2 rows just like PICK & ODDS
        stake_val = float(bet.get("stake", 0) or 0)
        if show_stakes:
            stake_str = f"${stake_val:,.0f}" if stake_val > 0 else "Tracked"
            _draw_pill(
                draw,
                col_x_stake,
                ry0 + 16,
                stake_str,
                f_num,
                (15, 23, 42),
                TEXT_PRIMARY if stake_val > 0 else TEXT_SECONDARY,
                pad_x=12,
                pad_y=5,
                border_color=(45, 55, 72),
            )
            if res in ("Win", "Loss") and stake_val > 0:
                try:
                    od = float(bet.get("odds_decimal") or 1.91)
                except (ValueError, TypeError):
                    od = 1.91
                row_pnl = stake_val * (od - 1.0) if res == "Win" else -stake_val
                p_col = ACCENT_GREEN if row_pnl >= 0 else ACCENT_RED
                draw.text((col_x_stake + 4, ry0 + 50), f"P/L: {row_pnl:+,.0f}", font=f_meta, fill=p_col)
            else:
                draw.text((col_x_stake + 4, ry0 + 50), f"EV: {ev_val:+.1%}", font=f_meta, fill=ACCENT_GREEN if ev_val > 0 else TEXT_SECONDARY)
        else:
            edge_lbl = f"+{edge_pts:.1f} pts" if edge_pts > 0 else f"{ev_val:+.1%}"
            _draw_pill(
                draw,
                col_x_stake,
                ry0 + 16,
                edge_lbl,
                f_num,
                (16, 42, 32),
                ACCENT_GREEN,
                pad_x=12,
                pad_y=5,
                border_color=(34, 197, 94),
            )
            draw.text(
                (col_x_stake + 4, ry0 + 50),
                f"EV: {ev_val:+.1%}",
                font=f_meta,
                fill=ACCENT_GREEN if ev_val > 0 else TEXT_SECONDARY,
            )

        # Column 6: Status Pill + Actual Score (if settled)
        if res == "Win":
            _draw_pill(draw, col_x_status, ry0 + 16, "WIN ✓", f_badge, (20, 83, 45), ACCENT_GREEN, pad_x=16, pad_y=5, border_color=ACCENT_GREEN)
            if bet.get("actual_value") is not None:
                try:
                    act_v = float(bet["actual_value"])
                    draw.text((col_x_status + 4, ry0 + 50), f"Act: {act_v:.1f}", font=f_meta, fill=TEXT_SECONDARY)
                except (ValueError, TypeError):
                    pass
        elif res == "Loss":
            _draw_pill(draw, col_x_status, ry0 + 16, "LOSS ✗", f_badge, (127, 29, 29), ACCENT_RED, pad_x=16, pad_y=5, border_color=ACCENT_RED)
            if bet.get("actual_value") is not None:
                try:
                    act_v = float(bet["actual_value"])
                    draw.text((col_x_status + 4, ry0 + 50), f"Act: {act_v:.1f}", font=f_meta, fill=TEXT_SECONDARY)
                except (ValueError, TypeError):
                    pass
        elif res == "Push":
            _draw_pill(draw, col_x_status, ry0 + 16, "PUSH", f_badge, (30, 58, 138), ACCENT_BLUE, pad_x=16, pad_y=5)
            if bet.get("actual_value") is not None:
                try:
                    act_v = float(bet["actual_value"])
                    draw.text((col_x_status + 4, ry0 + 50), f"Act: {act_v:.1f}", font=f_meta, fill=TEXT_SECONDARY)
                except (ValueError, TypeError):
                    pass
        else:
            _draw_pill(draw, col_x_status, ry0 + 16, "PENDING", f_badge, (66, 32, 6), ACCENT_GOLD, pad_x=14, pad_y=5, border_color=ACCENT_GOLD)
            draw.text((col_x_status + 4, ry0 + 50), "Upcoming", font=f_meta, fill=TEXT_SECONDARY)

        ry += row_h + row_gap

    # -----------------------------------------------------------------------
    # 4. Footer
    # -----------------------------------------------------------------------
    fy = total_h - pad_outer - 28
    draw.text(
        (pad_outer + 8, fy),
        "NBA BetLab  •  Sweet-Spot Quant Engine  •  Graded vs Official Box Scores",
        font=f_footer,
        fill=TEXT_SECONDARY,
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def copy_png_bytes_to_windows_clipboard(png_bytes: bytes) -> Tuple[bool, str]:
    """
    Copies raw PNG image bytes directly onto the native Windows OS clipboard
    so the user can press Ctrl+V anywhere (Discord, Messenger, Telegram, X, etc.).
    """
    if os.name != "nt":
        return False, "Native OS clipboard copy is only supported on Windows."

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(png_bytes)
            tmp_path = tmp.name

        # Escape any single quotes in path for PowerShell
        safe_path = tmp_path.replace("'", "''")
        ps_cmd = (
            "Add-Type -AssemblyName System.Windows.Forms,System.Drawing; "
            f"[System.Windows.Forms.Clipboard]::SetImage([System.Drawing.Image]::FromFile('{safe_path}'))"
        )
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Sta", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode == 0:
            return True, "Copied PNG to Windows Clipboard! Press Ctrl+V in Discord/Messenger to paste."
        return False, f"Clipboard warning: {res.stderr.strip()}"
    except Exception as e:
        return False, f"Failed to copy to clipboard: {e}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def _recap_decimal_odds(bet: Dict[str, Any]) -> float:
    """Best-effort decimal odds for a tracked bet (decimal field first, then American)."""
    for key in ("odds_decimal", "odds"):
        raw = bet.get(key)
        if raw is None:
            continue
        try:
            val = float(raw)
        except (ValueError, TypeError):
            continue
        if 1.01 <= val <= 50.0:
            return val
        if abs(val) >= 100:
            return 1.0 + (val / 100.0) if val > 0 else 1.0 + (100.0 / abs(val))
    am = bet.get("odds_american")
    try:
        am = float(am)
        if abs(am) >= 100:
            return 1.0 + (am / 100.0) if am > 0 else 1.0 + (100.0 / abs(am))
    except (ValueError, TypeError):
        pass
    return 1.909


def _recap_units(bet: Dict[str, Any]) -> float:
    """Flat 1u staking: win pays (odds - 1), loss costs 1u, push is 0."""
    res = bet.get("result")
    if res == "Win":
        return _recap_decimal_odds(bet) - 1.0
    if res == "Loss":
        return -1.0
    return 0.0


def _recap_record(bets: List[Dict[str, Any]]) -> Tuple[int, int, int, float, float]:
    """Returns (wins, losses, pushes, win_rate_pct_excl_pushes, net_units)."""
    w = sum(1 for b in bets if b.get("result") == "Win")
    l = sum(1 for b in bets if b.get("result") == "Loss")
    p = sum(1 for b in bets if b.get("result") == "Push")
    wr = (w / (w + l)) * 100.0 if (w + l) else 0.0
    u = sum(_recap_units(b) for b in bets)
    return w, l, p, wr, u


_RECAP_MARKET_LABELS = {
    "RA": "R+A", "PA": "P+A", "PR": "P+R", "PRA": "PRA",
    "PTS": "PTS", "REB": "REB", "AST": "AST", "3PT": "3PM",
}
_RECAP_MARKET_LEGEND = {
    "RA": "R+A = REB+AST", "PA": "P+A = PTS+AST",
    "PR": "P+R = PTS+REB", "PRA": "PRA = PTS+REB+AST",
}


def _recap_fmt_num(v: Any) -> str:
    try:
        f = float(v)
    except (ValueError, TypeError):
        return str(v)
    return f"{int(f)}" if f.is_integer() else f"{f:.1f}"


def _recap_pick_text(bet: Dict[str, Any]) -> Tuple[str, str]:
    """Returns (headline, detail) for a pick row."""
    mkt = str(bet.get("market", "")).upper()
    side = str(bet.get("side", "")).upper()
    actual = bet.get("actual_value")
    if mkt == "TOTAL":
        ou = "U" if "UNDER" in side else "O"
        detail = f"Total {ou}{_recap_fmt_num(bet.get('line'))}"
        if actual is not None:
            detail += f" · actual {_recap_fmt_num(actual)}"
        return str(bet.get("player", "")), detail
    if mkt == "SPREAD":
        return str(bet.get("player", "")), f"Spread {bet.get('side', '')}"
    ou = "U" if "UNDER" in side else "O"
    lbl = _RECAP_MARKET_LABELS.get(mkt, mkt)
    detail = f"{lbl} {ou}{_recap_fmt_num(bet.get('line'))}"
    if actual is not None:
        detail += f" · actual {_recap_fmt_num(actual)}"
    return str(bet.get("player", "")), detail


def _recap_fit_legend(label: str, markets: List[str], measure, max_w: float) -> str:
    """Append market legend entries (e.g. 'R+A = REB+AST') only while they fit max_w."""
    for m in markets:
        cand = label + "  ·  " + _RECAP_MARKET_LEGEND[m]
        if measure(cand) > max_w:
            break
        label = cand
    return label


def _recap_short_date(d: str) -> str:
    try:
        dt = datetime.strptime(d, "%Y-%m-%d")
        return f"{dt.month}/{dt.day}"
    except Exception:
        return d


def _recap_title_dates(d_min: str, d_max: str) -> str:
    try:
        a = datetime.strptime(d_min, "%Y-%m-%d")
        b = datetime.strptime(d_max, "%Y-%m-%d")
    except Exception:
        return f"{d_min} – {d_max}"
    if a == b:
        return f"{a.strftime('%b')} {a.day}"
    if (a.year, a.month) == (b.year, b.month):
        return f"{a.strftime('%b')} {a.day}–{b.day}"
    return f"{a.strftime('%b')} {a.day} – {b.strftime('%b')} {b.day}"


def render_performance_recap_png(
    bets: List[Dict[str, Any]],
    period_title: str = "WEEKLY PERFORMANCE RECAP",
    period_subtitle: Optional[str] = None,
    currency: str = "PHP",
    currency_symbol: str = "₱",
    hypothetical_stake: float = 100.0,
    width: int = 1024,
    context_bets: Optional[List[Dict[str, Any]]] = None,
    category_label: str = "Player Props",
    brand: str = "NBA BETLAB",
    scale: int = 2,
    max_picks: int = 16,
    small_sample_threshold: int = 30,
) -> bytes:
    """
    Renders the BetLab performance recap card.

    Layout: brand + title, 4 KPI tiles (record, win rate vs real breakeven, net
    units, ROI), a context strip (this period vs last 7 days vs tracked-to-date),
    every pick with line / actual / units, a daily net-units bar chart, and a
    one-line hypothetical stake simulation.

    `bets` is the recap window. `context_bets` is the full decided pool for the
    same category (used for the context strip and the daily chart); defaults to
    `bets`. `period_title` / `period_subtitle` are kept for backward compatibility
    but the title is now derived from the actual bet dates.
    Rendered at `scale`x for crisp text after messenger/mobile compression.
    """
    S = max(1, int(scale))

    def s(v: float) -> int:
        return int(round(v * S))

    decided = [b for b in bets if b.get("result") in ("Win", "Loss", "Push")]
    decided.sort(key=lambda b: (b.get("date", ""), b.get("tracked_at", "")))
    ctx = [b for b in (context_bets if context_bets is not None else bets)
           if b.get("result") in ("Win", "Loss", "Push")]

    wins, losses, pushes, win_rate, net_units = _recap_record(decided)
    graded = wins + losses
    staked = float(graded)
    roi_pct = (net_units / staked) * 100.0 if staked else 0.0
    wl_bets = [b for b in decided if b.get("result") in ("Win", "Loss")]
    avg_odds = (sum(_recap_decimal_odds(b) for b in wl_bets) / len(wl_bets)) if wl_bets else 1.909
    breakeven = 100.0 / avg_odds

    dates = sorted({b.get("date", "") for b in decided if b.get("date")})
    d_min = dates[0] if dates else datetime.now().strftime("%Y-%m-%d")
    d_max = dates[-1] if dates else d_min
    title = f"{_recap_title_dates(d_min, d_max)} Recap · {category_label}"

    # ---- Context strip ----------------------------------------------------
    ctx_upto = [b for b in ctx if b.get("date", "") <= d_max]
    try:
        end_dt = datetime.strptime(d_max, "%Y-%m-%d")
        l7_start = (end_dt - timedelta(days=6)).strftime("%Y-%m-%d")
    except Exception:
        l7_start = d_min
    l7 = [b for b in ctx_upto if b.get("date", "") >= l7_start]
    first_date = min((b.get("date", "") for b in ctx_upto if b.get("date")), default=d_min)

    pw, pl, _, _, pu = _recap_record(decided)
    l7w, l7l, _, l7wr, l7u = _recap_record(l7)
    tdw, tdl, _, tdwr, tdu = _recap_record(ctx_upto)
    period_lbl = "TODAY" if d_min == d_max else "THIS PERIOD"

    # ---- Daily chart data -------------------------------------------------
    by_day: Dict[str, float] = {}
    for b in ctx_upto:
        d = b.get("date", "")
        if d:
            by_day[d] = by_day.get(d, 0.0) + _recap_units(b)
    chart_days = sorted(by_day.keys())[-14:]

    # ---- Palette (brighter secondary text survives JPEG compression) -------
    C_BG = (10, 14, 22)
    C_CARD = (17, 24, 39)
    C_CARD_BORDER = (44, 58, 82)
    C_TILE = (22, 31, 48)
    C_TILE_BORDER = (50, 66, 92)
    C_WHITE = (238, 243, 250)
    C_MUTED = (172, 184, 202)
    C_LABEL = (150, 165, 188)
    C_CYAN = (56, 189, 248)
    C_GREEN = (52, 211, 153)
    C_GREEN_DIM = (31, 122, 92)
    C_RED = (248, 113, 113)
    C_RED_DIM = (140, 52, 56)
    C_PURPLE = (167, 139, 250)
    C_GOLD = (250, 204, 21)

    # ---- Fonts --------------------------------------------------------------
    f_brand = _load_font(s(14), bold=True)
    f_title = _load_font(s(34), bold=True)
    f_sub = _load_font(s(17))
    f_lbl = _load_font(s(13), bold=True)
    f_kpi = _load_font(s(34), bold=True, mono=True)
    f_kpi_sub = _load_font(s(14))
    f_ctx_lbl = _load_font(s(13), bold=True)
    f_ctx_val = _load_font(s(18), bold=True, mono=True)
    f_ctx_rest = _load_font(s(16), mono=True)
    f_pick_name = _load_font(s(16), bold=True)
    f_pick_det = _load_font(s(13))
    f_pick_u = _load_font(s(14), bold=True, mono=True)
    f_axis = _load_font(s(11))
    f_small = _load_font(s(13))
    f_hypo = _load_font(s(15))
    f_hypo_b = _load_font(s(15), bold=True)
    f_pill = _load_font(s(14))

    # ---- Geometry -------------------------------------------------------------
    W = width
    M = 26              # outer margin
    PX = 58             # inner content x
    RX = W - PX         # inner content right edge
    kpi_y, kpi_h = 166, 120
    ctx_y, ctx_h = 300, 70
    sec_y = 388
    grid_y = 412
    row_h, row_gap = 56, 8
    grid_x0, grid_x1 = PX, 642
    col_gap = 14
    col_w = (grid_x1 - grid_x0 - col_gap) / 2.0
    panel_x0, panel_x1 = 659, RX
    chart_y0, chart_y1 = grid_y, grid_y + 172

    # Single day: chronological. Multi-day: most recent first when truncated.
    pick_order = decided if d_min == d_max else sorted(
        decided, key=lambda b: (b.get("date", ""), b.get("tracked_at", "")), reverse=True)
    shown = pick_order[:max_picks]
    overflow = len(decided) - len(shown)
    slots = len(shown) + (1 if overflow > 0 else 0)
    n_rows = max(1, (slots + 1) // 2)
    grid_bottom = grid_y + n_rows * row_h + (n_rows - 1) * row_gap
    content_bottom = max(grid_bottom, chart_y1 + 40)
    footer_line_y = content_bottom + 18
    footer_text_y = footer_line_y + 14
    H = footer_text_y + 22 + 28 + M

    img = Image.new("RGB", (s(W), s(H)), color=C_BG)
    draw = ImageDraw.Draw(img)

    def text(x, y, t, font, fill):
        draw.text((s(x), s(y)), t, font=font, fill=fill)

    def text_w(t, font) -> float:
        bb = draw.textbbox((0, 0), t, font=font)
        return (bb[2] - bb[0]) / S

    def rrect(x0, y0, x1, y1, r, fill, outline=None, w=1):
        draw.rounded_rectangle([s(x0), s(y0), s(x1), s(y1)], radius=s(r), fill=fill,
                               outline=outline, width=s(w) if outline else 0)

    # Card + accent stripe
    rrect(M, M, W - M, H - M, 22, C_CARD, C_CARD_BORDER, 1)
    draw.rounded_rectangle([s(M + 16), s(M + 1), s(W - M - 16), s(M + 6)], radius=s(3), fill=C_CYAN)

    # Brand (letter-spaced)
    bx = PX
    for ch in brand:
        text(bx, 56, ch, f_brand, C_CYAN)
        bx += text_w(ch, f_brand) + 2.6

    text(PX, 76, title, f_title, C_WHITE)
    sub = f"{graded} settled bets · flat 1u staking · avg odds {avg_odds:.2f} (breakeven {breakeven:.1f}%)"
    text(PX, 122, sub, f_sub, C_MUTED)

    # Small-sample pill
    if graded < small_sample_threshold:
        pill_t = f"Small sample · {graded} bets"
        pw_ = text_w(pill_t, f_pill) + 30
        px0 = RX - pw_
        rrect(px0, 56, RX, 87, 15, (58, 46, 10), (133, 104, 20), 1)
        text(px0 + 15, 62, pill_t, f_pill, C_GOLD)

    # ---- KPI tiles -----------------------------------------------------------
    kgap = 14
    kw = (RX - PX - 3 * kgap) / 4.0
    pos = lambda v: C_GREEN if v >= 0 else C_RED  # noqa: E731
    kpis = [
        ("RECORD", f"{wins} - {losses}", f"{pushes} push" + ("" if pushes == 1 else "es"), C_WHITE),
        ("WIN RATE", f"{win_rate:.1f}%", f"vs {breakeven:.1f}% breakeven", C_GREEN if win_rate >= breakeven else C_RED),
        ("NET UNITS", f"{net_units:+.2f}u", f"{staked:.0f}u staked", pos(net_units)),
        ("ROI", f"{roi_pct:+.1f}%", f"on {graded} bets", pos(roi_pct)),
    ]
    for i, (lbl, val, subt, col) in enumerate(kpis):
        x0 = PX + i * (kw + kgap)
        rrect(x0, kpi_y, x0 + kw, kpi_y + kpi_h, 12, C_TILE, C_TILE_BORDER, 1)
        text(x0 + 19, kpi_y + 16, lbl, f_lbl, C_LABEL)
        text(x0 + 19, kpi_y + 36, val, f_kpi, col)
        text(x0 + 19, kpi_y + 86, subt, f_kpi_sub, C_MUTED)

    # ---- Context strip -------------------------------------------------------
    cgap = 14
    cw = (RX - PX - 2 * cgap) / 3.0
    ctx_cards = [
        (period_lbl, f"{pw}-{pl}", f" · {pu:+.2f}u", C_CYAN),
        (f"LAST 7 DAYS ({_recap_short_date(l7_start)}–{_recap_short_date(d_max)})",
         f"{l7w}-{l7l}", f" · {l7wr:.1f}% · {l7u:+.2f}u", C_GREEN),
        (f"TRACKED TO DATE (SINCE {_recap_short_date(first_date)})",
         f"{tdw}-{tdl}", f" · {tdwr:.1f}% · {tdu:+.2f}u", C_PURPLE),
    ]
    for i, (lbl, rec, rest, acc) in enumerate(ctx_cards):
        x0 = PX + i * (cw + cgap)
        rrect(x0, ctx_y, x0 + cw, ctx_y + ctx_h, 10, (20, 28, 44))
        draw.rounded_rectangle([s(x0), s(ctx_y), s(x0 + 4), s(ctx_y + ctx_h)], radius=s(2), fill=acc)
        text(x0 + 19, ctx_y + 14, lbl, f_ctx_lbl, C_LABEL)
        text(x0 + 19, ctx_y + 36, rec, f_ctx_val, C_WHITE)
        text(x0 + 19 + text_w(rec, f_ctx_val), ctx_y + 38, rest, f_ctx_rest, C_MUTED)

    # ---- Every pick ----------------------------------------------------------
    present = []
    for b in decided:
        m = str(b.get("market", "")).upper()
        if m in _RECAP_MARKET_LEGEND and m not in present:
            present.append(m)
    sec_lbl = "EVERY PICK" if not overflow else f"PICKS ({len(shown)} OF {len(decided)})"
    if present:
        sec_lbl = _recap_fit_legend(sec_lbl, present, lambda t: text_w(t, f_lbl), grid_x1 - grid_x0)
    text(PX, sec_y, sec_lbl, f_lbl, C_LABEL)

    def draw_icon(cx, cy, result):
        r = 11
        col = C_GREEN if result == "Win" else (C_RED if result == "Loss" else C_LABEL)
        draw.ellipse([s(cx - r), s(cy - r), s(cx + r), s(cy + r)], fill=col)
        ink = (12, 20, 30)
        lw = max(2, s(2.4))
        if result == "Win":
            draw.line([(s(cx - 5), s(cy + 0.5)), (s(cx - 1.5), s(cy + 4)), (s(cx + 5.5), s(cy - 4))],
                      fill=ink, width=lw, joint="curve")
        elif result == "Loss":
            draw.line([(s(cx - 4.5), s(cy - 4.5)), (s(cx + 4.5), s(cy + 4.5))], fill=ink, width=lw)
            draw.line([(s(cx - 4.5), s(cy + 4.5)), (s(cx + 4.5), s(cy - 4.5))], fill=ink, width=lw)
        else:
            draw.line([(s(cx - 5), s(cy)), (s(cx + 5), s(cy))], fill=ink, width=lw)

    for idx in range(slots):
        col_i = idx % 2
        row_i = idx // 2
        x0 = grid_x0 + col_i * (col_w + col_gap)
        y0 = grid_y + row_i * (row_h + row_gap)
        x1 = x0 + col_w
        rrect(x0, y0, x1, y0 + row_h, 9, (20, 28, 44), C_TILE_BORDER, 1)
        if idx >= len(shown):
            text(x0 + 16, y0 + 18, f"+{overflow} more picks", f_pick_name, C_MUTED)
            continue
        b = shown[idx]
        res = b.get("result")
        draw_icon(x0 + 24, y0 + row_h / 2, res)
        name, det = _recap_pick_text(b)
        u = _recap_units(b)
        u_txt = f"{u:+.2f}u" if res != "Push" else "push"
        u_col = C_GREEN if u > 0 else (C_RED if u < 0 else C_LABEL)
        u_w = text_w(u_txt, f_pick_u)
        max_name_w = col_w - 44 - u_w - 20
        if text_w(name, f_pick_name) > max_name_w:
            while len(name) > 1 and text_w(name + "…", f_pick_name) > max_name_w:
                name = name[:-1]
            name = name.rstrip() + "…"
        text(x0 + 44, y0 + 8, name, f_pick_name, C_WHITE)
        text(x0 + 44, y0 + 31, det, f_pick_det, C_MUTED)
        text(x1 - 12 - u_w, y0 + 19, u_txt, f_pick_u, u_col)

    # ---- Daily net units chart ----------------------------------------------
    text(panel_x0, sec_y, "DAILY NET UNITS", f_lbl, C_LABEL)
    rrect(panel_x0, chart_y0, panel_x1, chart_y1, 10, (20, 28, 44), C_TILE_BORDER, 1)
    if chart_days:
        top, bot = chart_y0 + 18, chart_y1 - 34
        mid = (top + bot) / 2.0
        half = (bot - top) / 2.0
        peak = max(abs(by_day[d]) for d in chart_days) or 1.0
        draw.line([(s(panel_x0 + 14), s(mid)), (s(panel_x1 - 14), s(mid))], fill=(60, 76, 102), width=max(1, S))
        inner = (panel_x1 - panel_x0) - 28
        slot = inner / len(chart_days)
        bar_w = min(22.0, slot * 0.68)
        recap_days = set(dates)
        for i, d in enumerate(chart_days):
            v = by_day[d]
            cx = panel_x0 + 14 + slot * (i + 0.5)
            h = max(3.0, abs(v) / peak * half)
            hi = d in recap_days
            if v >= 0:
                col = C_GREEN if hi else C_GREEN_DIM
                y_a, y_b = mid - h, mid
            else:
                col = C_RED if hi else C_RED_DIM
                y_a, y_b = mid, mid + h
            draw.rounded_rectangle([s(cx - bar_w / 2), s(y_a), s(cx + bar_w / 2), s(y_b)], radius=s(2), fill=col)
            day_lbl = d[-2:].lstrip("0") or d[-2:]
            text(cx - text_w(day_lbl, f_axis) / 2, chart_y1 - 26, day_lbl, f_axis, C_LABEL)

    # Hypothetical stake (one small line)
    profit = net_units * hypothetical_stake
    sign = "+" if profit >= 0 else "−"
    part_a = f"{currency_symbol}{hypothetical_stake:,.0f}/bet → "
    part_b = f"{sign}{currency_symbol}{abs(profit):,.0f}"
    part_c = " (hypothetical)"
    total_w = text_w(part_a, f_hypo) + text_w(part_b, f_hypo_b) + text_w(part_c, f_hypo)
    hx = (panel_x0 + panel_x1) / 2.0 - total_w / 2.0
    hy = chart_y1 + 14
    text(hx, hy, part_a, f_hypo, C_MUTED)
    hx += text_w(part_a, f_hypo)
    text(hx, hy, part_b, f_hypo_b, C_WHITE)
    hx += text_w(part_b, f_hypo_b)
    text(hx, hy, part_c, f_hypo, C_MUTED)

    # ---- Footer ----------------------------------------------------------------
    draw.line([(s(PX), s(footer_line_y)), (s(RX), s(footer_line_y))], fill=(44, 58, 82), width=max(1, S))
    scope = {"Player Props": "Props only", "Game Lines": "Game lines only"}.get(category_label, "All markets")
    text(PX, footer_text_y, f"{scope} · pushes excluded · graded vs official box scores", f_small, C_LABEL)
    gen = f"Generated {datetime.now().strftime('%b')} {datetime.now().day}, {datetime.now().year}"
    text(RX - text_w(gen, f_small), footer_text_y, gen, f_small, C_LABEL)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

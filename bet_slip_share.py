"""
bet_slip_share.py — Generates a sleek, dark-mode shareable PNG card of tracked bets
(Game Lines & Player Props) and copies it directly to the Windows clipboard.
"""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
from datetime import datetime
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
        "NBA Prop & Game Analyzer  •  Sweet-Spot Quant Engine  •  Closing Line Value (CLV) Verified",
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


def render_performance_recap_png(
    bets: List[Dict[str, Any]],
    period_title: str = "WEEKLY PERFORMANCE RECAP",
    period_subtitle: Optional[str] = None,
    currency: str = "PHP",
    currency_symbol: str = "₱",
    hypothetical_stake: float = 1000.0,
    width: int = 880,
) -> bytes:
    """
    Renders a high-level summary recap card (Record, Win Rate, ROI, Units, and
    Hypothetical Fiat Currency Simulation) without listing individual bets.
    Returns raw PNG bytes.
    """
    # 1. Filter decided bets
    decided = [b for b in bets if b.get("result") in ("Win", "Loss", "Push")]
    wins = sum(1 for b in decided if b.get("result") == "Win")
    losses = sum(1 for b in decided if b.get("result") == "Loss")
    pushes = sum(1 for b in decided if b.get("result") == "Push")
    total_decided = len(decided)
    win_rate = (wins / (wins + losses)) * 100.0 if (wins + losses) > 0 else 0.0

    # Units & ROI calculation
    net_units = 0.0
    for b in decided:
        res = b.get("result")
        try:
            od = float(b.get("odds_decimal") or b.get("odds") or 1.909)
            if od < 1.01 or od > 50.0:
                am = int(od)
                od = 1.0 + (am / 100.0) if am > 0 else 1.0 + (100.0 / abs(am))
        except (ValueError, TypeError):
            od = 1.909
        if res == "Win":
            net_units += (od - 1.0)
        elif res == "Loss":
            net_units -= 1.0

    total_staked_units = total_decided * 1.0
    roi_pct = (net_units / total_staked_units) * 100.0 if total_staked_units > 0 else 0.0
    hypo_staked = total_staked_units * hypothetical_stake
    hypo_profit = net_units * hypothetical_stake

    # Market segments
    def _seg_stats(seg_bets: List[Dict[str, Any]]) -> Tuple[int, int, int, float, float]:
        w = sum(1 for b in seg_bets if b.get("result") == "Win")
        l = sum(1 for b in seg_bets if b.get("result") == "Loss")
        p = sum(1 for b in seg_bets if b.get("result") == "Push")
        wr = (w / (w + l)) * 100.0 if (w + l) > 0 else 0.0
        u = 0.0
        for b in seg_bets:
            try:
                od = float(b.get("odds_decimal") or b.get("odds") or 1.909)
                if od < 1.01 or od > 50.0:
                    am = int(od)
                    od = 1.0 + (am / 100.0) if am > 0 else 1.0 + (100.0 / abs(am))
            except (ValueError, TypeError):
                od = 1.909
            if b.get("result") == "Win":
                u += (od - 1.0)
            elif b.get("result") == "Loss":
                u -= 1.0
        return w, l, p, wr, u

    prop_bets = [b for b in decided if not _is_game_bet(b)]
    spread_bets = [b for b in decided if b.get("market") == "SPREAD" or b.get("bet_type") == "spread"]
    total_bets = [b for b in decided if b.get("market") == "TOTAL" or b.get("bet_type") == "total"]

    pw, pl, pp, pwr, pu = _seg_stats(prop_bets)
    sw, sl, sp, swr, su = _seg_stats(spread_bets)
    tw, tl, tp, twr, tu = _seg_stats(total_bets)

    # Dimensions
    pad_outer = 26
    content_w = width - (2 * pad_outer)
    header_h = 114
    kpi_h = 96
    hypo_h = 124
    seg_h = 92
    footer_h = 32
    gap = 14

    total_h = (
        pad_outer
        + header_h
        + gap
        + kpi_h
        + gap
        + hypo_h
        + gap
        + seg_h
        + gap
        + footer_h
        + pad_outer
    )

    img = Image.new("RGB", (width, total_h), color=BG_DARK)
    draw = ImageDraw.Draw(img)

    # Fonts
    f_title = _load_font(24, bold=True)
    f_sub = _load_font(14, bold=False)
    f_kpi_lbl = _load_font(12, bold=True)
    f_kpi_val = _load_font(24, bold=True, mono=True)
    f_kpi_sub = _load_font(12, bold=False)
    f_hypo_main = _load_font(21, bold=True, mono=True)
    f_hypo_sub = _load_font(13, bold=False)
    f_seg_title = _load_font(13, bold=True)
    f_seg_val = _load_font(17, bold=True, mono=True)
    f_seg_sub = _load_font(12, bold=False)
    f_badge = _load_font(12, bold=True)
    f_footer = _load_font(12, bold=False)

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
    # Accent top stripe
    draw.rounded_rectangle(
        [hx0 + 2, hy0 + 2, hx1 - 2, hy0 + 8],
        radius=4,
        fill=ACCENT_CYAN,
    )

    # Title & Subtitle
    draw.text((hx0 + 24, hy0 + 20), period_title, font=f_title, fill=TEXT_PRIMARY)
    now_str = datetime.now().strftime("%b %d, %Y • %I:%M %p")
    sub_text = period_subtitle or f"Model Performance Recap • Generated {now_str}"
    draw.text((hx0 + 24, hy0 + 54), sub_text, font=f_sub, fill=TEXT_SECONDARY)

    # Pills
    px = hx0 + 24
    py = hy0 + 78
    px += _draw_pill(draw, px, py, f"DECIDED PLAYS: {total_decided}", f_badge, (30, 41, 59), ACCENT_CYAN, border_color=ACCENT_CYAN) + 10
    px += _draw_pill(draw, px, py, f"PROPS: {len(prop_bets)}", f_badge, (59, 7, 100), ACCENT_PURPLE) + 10
    px += _draw_pill(draw, px, py, f"GAME LINES: {len(spread_bets) + len(total_bets)}", f_badge, (23, 37, 84), ACCENT_BLUE) + 10

    # -----------------------------------------------------------------------
    # 2. Top KPI Stat Boxes (4 Cards)
    # -----------------------------------------------------------------------
    kpi_y0 = hy1 + gap
    kpi_gap = 12
    kpi_w = (content_w - (3 * kpi_gap)) // 4

    kpi_configs = [
        ("RECORD (W-L)", f"{wins} - {losses}", f"{pushes} Push" if pushes else f"{total_decided} Decided", TEXT_PRIMARY),
        ("WIN RATE", f"{win_rate:.1f}%", "BE: 52.4% (-110)", ACCENT_GREEN if win_rate >= 52.4 else ACCENT_RED),
        ("NET UNITS (±)", f"{net_units:+.2f}u", "Flat 1.0u Staking", ACCENT_GREEN if net_units >= 0 else ACCENT_RED),
        ("REALIZED ROI", f"{roi_pct:+.1f}%", f"{total_staked_units:.0f}u Total Staked", ACCENT_GREEN if roi_pct >= 0 else ACCENT_RED),
    ]

    for i, (lbl, val, sub, val_col) in enumerate(kpi_configs):
        kx0 = pad_outer + i * (kpi_w + kpi_gap)
        kx1 = kx0 + kpi_w
        ky0 = kpi_y0
        ky1 = ky0 + kpi_h

        draw.rounded_rectangle(
            [kx0, ky0, kx1, ky1],
            radius=12,
            fill=CARD_BG,
            outline=BORDER_SUBTLE,
            width=1,
        )
        draw.text((kx0 + 16, ky0 + 14), lbl, font=f_kpi_lbl, fill=TEXT_SECONDARY)
        draw.text((kx0 + 16, ky0 + 36), val, font=f_kpi_val, fill=val_col)
        draw.text((kx0 + 16, ky0 + 70), sub, font=f_kpi_sub, fill=TEXT_SECONDARY)

    # -----------------------------------------------------------------------
    # 3. Hypothetical Currency Simulation Container
    # -----------------------------------------------------------------------
    hypo_y0 = kpi_y0 + kpi_h + gap
    hypo_x0 = pad_outer
    hypo_x1 = width - pad_outer
    hypo_y1 = hypo_y0 + hypo_h

    # Container with subtle gradient border
    draw.rounded_rectangle(
        [hypo_x0, hypo_y0, hypo_x1, hypo_y1],
        radius=14,
        fill=(18, 25, 38),
        outline=(65, 80, 110),
        width=2,
    )
    # Left accent vertical bar
    bar_color = ACCENT_GREEN if hypo_profit >= 0 else ACCENT_RED
    draw.rounded_rectangle(
        [hypo_x0 + 3, hypo_y0 + 12, hypo_x0 + 8, hypo_y1 - 12],
        radius=3,
        fill=bar_color,
    )

    # Top pill
    _draw_pill(
        draw,
        hypo_x0 + 22,
        hypo_y0 + 14,
        "💰 HYPOTHETICAL STAKE SIMULATION",
        f_badge,
        (40, 30, 15) if hypo_profit >= 0 else (45, 20, 20),
        ACCENT_GOLD if hypo_profit >= 0 else ACCENT_RED,
        border_color=(120, 95, 30) if hypo_profit >= 0 else (120, 45, 45),
        pad_x=10,
        pad_y=3,
    )

    # Big Profit Headline
    sign = "+" if hypo_profit >= 0 else "-"
    abs_profit = abs(hypo_profit)
    profit_text = f"{sign}{currency_symbol}{abs_profit:,.0f} {currency}"
    sim_headline = f"If put {currency_symbol}{hypothetical_stake:,.0f} {currency} / bet  →  Net Profit: {profit_text}"
    draw.text((hypo_x0 + 22, hypo_y0 + 46), sim_headline, font=f_hypo_main, fill=bar_color)

    # Subtext details
    sub_sim = (
        f"Total Staked: {currency_symbol}{hypo_staked:,.0f} {currency} on {total_decided} bets   •   "
        f"Return on Capital: {roi_pct:+.1f}%   •   Net Gain: {net_units:+.2f} Units"
    )
    draw.text((hypo_x0 + 22, hypo_y0 + 84), sub_sim, font=f_hypo_sub, fill=TEXT_SECONDARY)

    # -----------------------------------------------------------------------
    # 4. Market Segments Breakdown (3 Cards)
    # -----------------------------------------------------------------------
    seg_y0 = hypo_y1 + gap
    seg_y1 = seg_y0 + seg_h
    seg_gap = 12
    seg_w = (content_w - (2 * seg_gap)) // 3

    seg_data = [
        ("👤 PLAYER PROPS", f"{pw}W - {pl}L ({pwr:.1f}%)", f"{pu:+.2f} Units", ACCENT_PURPLE, pu >= 0),
        ("🏀 GAME SPREADS", f"{sw}W - {sl}L ({swr:.1f}%)", f"{su:+.2f} Units", ACCENT_BLUE, su >= 0),
        ("🎯 GAME TOTALS", f"{tw}W - {tl}L ({twr:.1f}%)", f"{tu:+.2f} Units", ACCENT_CYAN, tu >= 0),
    ]

    for i, (title, rec_str, unit_str, badge_col, is_pos) in enumerate(seg_data):
        sx0 = pad_outer + i * (seg_w + seg_gap)
        sx1 = sx0 + seg_w
        sy0 = seg_y0
        sy1 = sy0 + seg_h

        draw.rounded_rectangle(
            [sx0, sy0, sx1, sy1],
            radius=12,
            fill=CARD_BG,
            outline=BORDER_SUBTLE,
            width=1,
        )
        draw.text((sx0 + 16, sy0 + 14), title, font=f_seg_title, fill=badge_col)
        draw.text((sx0 + 16, sy0 + 36), rec_str, font=f_seg_val, fill=TEXT_PRIMARY)
        unit_col = ACCENT_GREEN if is_pos else ACCENT_RED
        draw.text((sx0 + 16, sy0 + 64), f"Profit: {unit_str}", font=f_seg_sub, fill=unit_col)

    # -----------------------------------------------------------------------
    # 5. Footer
    # -----------------------------------------------------------------------
    fy = seg_y1 + gap + 10
    draw.text(
        (pad_outer + 4, fy),
        "NBA Quant Model  •  Graded Against Official Box Scores  •  Closing Line Value (CLV) Verified",
        font=f_footer,
        fill=TEXT_SECONDARY,
    )
    ts_footer = f"Generated {datetime.now().strftime('%b %d, %Y')}"
    draw.text(
        (width - pad_outer - 150, fy),
        ts_footer,
        font=f_footer,
        fill=TEXT_SECONDARY,
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

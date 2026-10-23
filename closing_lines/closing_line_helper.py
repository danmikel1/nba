"""python closing_line_cli.pyclosing_line_helper.py — Batch Closing Line Entry Tool

Streamlit app to quickly populate closing_line and clv fields on bets.
Replaces manual JSON editing with a fast UI.

Usage:
    streamlit run closing_line_helper.py

Speed-ups:
• Shows open bets (closing_line missing or 0)
• Quick entry form + optional CSV upload
• Auto-calculate CLV
• One-click save back to bet_tracker.json
"""

import streamlit as st
import pandas as pd
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# ============================================================================
# Config
# ============================================================================
TRACKER_PATH = Path(__file__).resolve().parent.parent / "data" / "bet_tracker.json"


def load_tracker() -> List[Dict]:
    """Load bet_tracker.json."""
    if not TRACKER_PATH.exists():
        st.error(f"Tracker not found: {TRACKER_PATH}")
        return []
    with open(TRACKER_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_tracker(data: List[Dict]) -> bool:
    """Save bet_tracker.json."""
    try:
        with open(TRACKER_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False


def calculate_clv(opening_line: float, closing_line: float, side: str) -> float:
    """
    Calculate Closing Line Value (CLV) in points.
    CLV > 0 means market moved in your favor.
    CLV = (closing_line - opening_line) if OVER, else -(closing_line - opening_line)
    """
    if pd.isna(opening_line) or pd.isna(closing_line):
        return 0.0
    delta = closing_line - opening_line
    if side.upper() in ['UNDER', 'U']:
        delta = -delta
    return round(float(delta), 2)


def get_open_bets(data: List[Dict]) -> pd.DataFrame:
    """Filter bets with missing or zero closing_line (not yet settled)."""
    df = pd.DataFrame(data)
    if len(df) == 0:
        return df
    
    # Convert to numeric, handle nulls
    df['closing_line'] = pd.to_numeric(df.get('closing_line', [None]*len(df)), errors='coerce').fillna(0)
    df['clv'] = pd.to_numeric(df.get('clv', [None]*len(df)), errors='coerce').fillna(0)
    df['id'] = df.get('id', [None]*len(df))
    
    # Open bets: closing_line is None, 0, or not set
    open_mask = (df['closing_line'].isna()) | (df['closing_line'] == 0)
    open_bets = df[open_mask].copy()
    
    if len(open_bets) == 0:
        return pd.DataFrame()
    
    # Sort by date (newest first)
    open_bets['date'] = pd.to_datetime(open_bets.get('date', ''), errors='coerce')
    open_bets = open_bets.sort_values('date', ascending=False)
    
    return open_bets


def main():
    st.set_page_config(page_title="Closing Line Helper", layout="wide")
    st.title("⚡ Closing Line Entry Tool")
    st.write("Quickly populate **closing_line** and **CLV** for your open bets.")
    
    # Load data
    tracker_data = load_tracker()
    if not tracker_data:
        st.error("No bet tracker data found.")
        return
    
    open_bets = get_open_bets(tracker_data)
    
    if len(open_bets) == 0:
        st.success("✅ All bets have closing lines!")
        return
    
    # Tab 1: Manual Entry (Single Bet)
    tab1, tab2, tab3 = st.tabs(["📝 Single Entry", "📥 CSV Batch Upload", "📊 Summary"])
    
    # ========== TAB 1: MANUAL ENTRY ==========
    with tab1:
        st.subheader(f"Open Bets: {len(open_bets)}")
        
        # Show a few open bets as reference
        display_cols = ['date', 'player', 'line', 'side', 'market', 'odds', 'prob', 'tag']
        display_cols = [c for c in display_cols if c in open_bets.columns]
        st.dataframe(
            open_bets[display_cols].head(10),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        st.write("### Select a bet and enter its closing line:")
        
        # Dropdown to pick a bet
        if 'id' in open_bets.columns and open_bets['id'].notna().any():
            bet_ids = open_bets[open_bets['id'].notna()]['id'].tolist()
            display_strs = [
                f"[{open_bets[open_bets['id']==bid].iloc[0].get('date', '')}] "
                f"{open_bets[open_bets['id']==bid].iloc[0].get('player', '')} "
                f"{open_bets[open_bets['id']==bid].iloc[0].get('market', '')}"
                for bid in bet_ids
            ]
            selected_display = st.selectbox("Choose bet:", display_strs)
            if selected_display:
                selected_idx = display_strs.index(selected_display)
                selected_id = bet_ids[selected_idx]
                selected_bet = open_bets[open_bets['id'] == selected_id].iloc[0]
            else:
                selected_bet = None
        else:
            # Fallback: by index
            idx = st.selectbox("Choose bet by row:", range(len(open_bets)))
            selected_bet = open_bets.iloc[idx] if idx is not None else None
        
        if selected_bet is not None:
            st.markdown(f"**Selected:** {selected_bet.get('player', 'N/A')} | "
                       f"Line: {selected_bet.get('line', 'N/A')} | "
                       f"Side: {selected_bet.get('side', 'N/A').upper()}")
            
            col1, col2 = st.columns(2)
            with col1:
                closing_line = st.number_input(
                    "Closing Line:",
                    value=float(selected_bet.get('line', 0)),
                    step=0.5,
                    key='closing_line_single'
                )
            with col2:
                st.info(f"Opening: {selected_bet.get('line', 'N/A')}")
            
            if st.button("✅ Save Closing Line"):
                # Find and update the bet in tracker_data
                for i, bet in enumerate(tracker_data):
                    if bet.get('id') == selected_bet.get('id') or \
                       (bet.get('player') == selected_bet.get('player') and
                        bet.get('line') == selected_bet.get('line') and
                        bet.get('date') == selected_bet.get('date')):
                        
                        tracker_data[i]['closing_line'] = closing_line
                        # Calculate CLV
                        clv_val = calculate_clv(
                            selected_bet.get('line'),
                            closing_line,
                            selected_bet.get('side', 'OVER')
                        )
                        tracker_data[i]['clv'] = clv_val
                        
                        if save_tracker(tracker_data):
                            st.success(f"✅ Saved! CLV: {clv_val:+.2f}")
                            st.rerun()
                        else:
                            st.error("Save failed.")
                        break
    
    # ========== TAB 2: CSV BATCH UPLOAD ==========
    with tab2:
        st.subheader("Batch Upload Closing Lines via CSV")
        st.write("""
        Upload a CSV with columns: **player, market, line, side, date, closing_line**
        (Only these are required; other columns ignored.)
        """)
        
        # Generate sample CSV for download
        sample_data = {
            'player': ['Luka Doncic', 'Jayson Tatum'],
            'market': ['PTS', 'REB'],
            'line': [28.5, 8.5],
            'side': ['OVER', 'OVER'],
            'date': ['2026-03-19', '2026-03-19'],
            'closing_line': [28.5, 9.0],
        }
        sample_df = pd.DataFrame(sample_data)
        st.download_button(
            "📥 Download Sample CSV",
            sample_df.to_csv(index=False),
            "closing_lines_sample.csv",
            "text/csv"
        )
        
        uploaded_file = st.file_uploader("Upload CSV:", type=['csv'])
        
        if uploaded_file:
            try:
                csv_df = pd.read_csv(uploaded_file)
                st.dataframe(csv_df, use_container_width=True)
                
                if st.button("🚀 Apply All Closing Lines"):
                    updated_count = 0
                    for _, row in csv_df.iterrows():
                        player = row.get('player', '').strip()
                        market = row.get('market', '').strip()
                        line = pd.to_numeric(row.get('line', np.nan), errors='coerce')
                        closing = pd.to_numeric(row.get('closing_line', np.nan), errors='coerce')
                        side = row.get('side', '').strip()
                        date_str = row.get('date', '').strip()
                        
                        if pd.isna(closing) or not player:
                            continue
                        
                        # Match bet in tracker_data
                        for i, bet in enumerate(tracker_data):
                            if (bet.get('player', '').strip().lower() == player.lower() and
                                bet.get('market', '').strip().upper() == market.upper() and
                                pd.to_numeric(bet.get('line', np.nan), errors='coerce') == line):
                                
                                tracker_data[i]['closing_line'] = closing
                                clv_val = calculate_clv(line, closing, side)
                                tracker_data[i]['clv'] = clv_val
                                updated_count += 1
                                break
                    
                    if save_tracker(tracker_data):
                        st.success(f"✅ Updated {updated_count} bets!")
                        st.rerun()
                    else:
                        st.error("Save failed.")
            except Exception as e:
                st.error(f"CSV parse error: {e}")
    
    # ========== TAB 3: SUMMARY ==========
    with tab3:
        st.subheader("Stats")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Open Bets", len(open_bets))
        with col2:
            closed = len(tracker_data) - len(open_bets)
            st.metric("Closed (with CLV)", closed)
        with col3:
            pct = 100 * closed / len(tracker_data) if len(tracker_data) > 0 else 0
            st.metric("% Complete", f"{pct:.0f}%")
        
        # Average CLV of closed bets
        df_all = pd.DataFrame(tracker_data)
        df_all['clv'] = pd.to_numeric(df_all.get('clv', 0), errors='coerce').fillna(0)
        closed_bets = df_all[df_all['clv'] != 0]
        if len(closed_bets) > 0:
            avg_clv = closed_bets['clv'].mean()
            st.metric("Avg CLV (closed)", f"{avg_clv:+.2f}")


if __name__ == "__main__":
    main()

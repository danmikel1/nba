#!/usr/bin/env python3
"""
Booker Recalibration Test Script
Run this to recalibrate the ML model for Devin Booker based on backtest results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nba_prediction import recalibrate_booker_model

if __name__ == "__main__":
    print("🔄 Starting Devin Booker model recalibration...")
    print("=" * 60)

    result = recalibrate_booker_model()

    print("\n" + "=" * 60)
    if result.get('success'):
        print("✅ Recalibration completed successfully!")
    else:
        print("❌ Recalibration failed!")
        print(f"Error: {result.get('error', 'Unknown error')}")
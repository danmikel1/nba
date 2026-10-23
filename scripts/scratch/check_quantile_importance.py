import joblib
import pandas as pd
import os
from pathlib import Path

# --- CONFIGURATION ---
DATA_DIR = Path("data")  # Adjust if your data folder is named differently
# By default discover all quantile model files in the data directory (per-market and group)
MODEL_GLOB = "nba_quantile_*.pkl"
# Prefer latest-feature-file (v21) but fall back to v20 for backward compatibility
PREFERRED_FEATURE_FILES = ["nba_features_v21.pkl", "nba_features_v20.pkl"]


def get_feature_names():
    """Load the feature names list used during training.

    Behavior:
    - prefer `nba_features_v21.pkl` if present
    - fall back to `nba_features_v20.pkl`
    - returns list of feature names or None
    """
    for fn in PREFERRED_FEATURE_FILES:
        p = DATA_DIR / fn
        if p.exists():
            try:
                features = joblib.load(p)
                print(f"Loaded feature list from: {p.name}")
                return features
            except Exception as e:
                print(f"❌ Failed to load {p.name}: {e}")
    print(f"❌ Could not find any feature file among: {PREFERRED_FEATURE_FILES}")
    return None


def discover_model_files():
    """Return sorted list of quantile model files to analyze.

    - Includes per-market files (e.g. PTS) and group files (scoring/counting/...) if present
    - Excludes backup/legacy filenames by pattern
    """
    files = sorted([p.name for p in (DATA_DIR).glob(MODEL_GLOB)])
    if not files:
        print(f"No quantile model files found with glob '{MODEL_GLOB}' in {DATA_DIR}")
    return files

def analyze_model(filename, feature_names):
    path = DATA_DIR / filename
    if not path.exists():
        return
    
    print(f"\n==========================================")
    print(f"🔍 ANALYZING: {filename}")
    print(f"==========================================")
    
    try:
        model_dict = joblib.load(path)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return

    # specific keys to check (V21.5 uses q05/q95 instead of q10/q90)
    # Try new keys first, fall back to old for backward compatibility
    quantiles_new = ['q05', 'q50', 'q95']
    quantiles_old = ['q10', 'q50', 'q90']
    
    # Detect which quantile scheme this model uses
    if 'q05' in model_dict or 'q95' in model_dict:
        quantiles = quantiles_new
        print("   Using V21.5 quantiles: q05/q50/q95")
    else:
        quantiles = quantiles_old
        print("   Using legacy quantiles: q10/q50/q90")
    
    # Determine actual feature count from model (may differ from feature_names
    # if features were added/removed since the feature list was saved)
    model_n_features = None
    for q in quantiles:
        if q in model_dict:
            m = model_dict[q]
            if hasattr(m, 'n_features_in_'):
                model_n_features = m.n_features_in_
                break
            elif hasattr(m, 'feature_importances_'):
                model_n_features = len(m.feature_importances_)
                break

    # Align feature names to model dimension
    if model_n_features is not None and model_n_features != len(feature_names):
        print(f"⚠️ Feature list has {len(feature_names)} names but model expects {model_n_features}.")
        if model_n_features < len(feature_names):
            # Model was trained after a feature was removed — find which ones the model doesn't have
            # Try to detect by checking if a known-removed feature is in the list
            aligned = [f for f in feature_names if f != 'feat_min'][:model_n_features]
            if len(aligned) == model_n_features:
                feature_names = aligned
                print(f"   Aligned to {len(feature_names)} features (dropped removed features).")
            else:
                feature_names = feature_names[:model_n_features]
                print(f"   Truncated feature list to {model_n_features}.")
        else:
            # Model has more features than list — pad with placeholders
            feature_names = list(feature_names) + [f'unknown_{i}' for i in range(model_n_features - len(feature_names))]
            print(f"   Padded feature list to {model_n_features}.")

    # DataFrame to hold all importances
    df_imp = pd.DataFrame({'feature': feature_names})
    
    valid_model_found = False
    
    for q in quantiles:
        if q not in model_dict:
            continue
            
        model = model_dict[q]
        valid_model_found = True
        
        # Get importances (works for XGBoost sklearn API)
        # importance_type='gain' is usually best for "Impact"
        # importance_type='weight' is "Frequency"
        try:
            # XGBRegressor stores this in feature_importances_
            imps = model.feature_importances_
            if len(imps) == len(feature_names):
                df_imp[f'{q}_importance'] = imps
            else:
                print(f"⚠️ {q} has {len(imps)} importances vs {len(feature_names)} features — skipping")
        except AttributeError:
            print(f"⚠️ Could not extract importances from {q} (Model type unknown)")

    if not valid_model_found:
        print("❌ No valid quantile models found in file.")
        return

    # Ensure all expected importance columns exist (missing quantiles -> zero importance)
    for q in quantiles:
        col = f'{q}_importance'
        if col not in df_imp.columns:
            df_imp[col] = 0.0

    # Calculate Average Importance (Mean of available importance columns)
    imp_cols = [c for c in df_imp.columns if c.endswith('_importance')]
    df_imp['avg_importance'] = df_imp[imp_cols].mean(axis=1)

    # Prefer sorting by q50 (median) when available and non-trivial, otherwise fallback to avg_importance
    if 'q50_importance' in df_imp.columns and df_imp['q50_importance'].abs().sum() > 0:
        sort_by = 'q50_importance'
    else:
        sort_by = 'avg_importance'

    df_imp = df_imp.sort_values(sort_by, ascending=False).reset_index(drop=True)

    # Display Top 15 (dynamically use whatever quantile columns exist)
    importance_display_cols = [f'{q}_importance' for q in quantiles if f'{q}_importance' in df_imp.columns]
    display_cols = ['feature'] + importance_display_cols
    print("\n🏆 TOP 15 DRIVERS (Sorted by {}):".format(sort_by))
    try:
        print(df_imp[display_cols].head(15).to_markdown(index=False, floatfmt=".4f"))
    except Exception:
        print(df_imp[display_cols].head(15).to_string(index=False))

    # Check for "Floor vs Ceiling" differences (handle NaN safely)
    # Use q95/q05 for new models, q90/q10 for legacy
    high_q = 'q95_importance' if 'q95_importance' in df_imp.columns else 'q90_importance'
    low_q = 'q05_importance' if 'q05_importance' in df_imp.columns else 'q10_importance'
    high_vals = df_imp[high_q] if high_q in df_imp.columns else 0.0
    low_vals = df_imp[low_q] if low_q in df_imp.columns else 0.0
    df_imp['volatility_driver'] = high_vals - low_vals

    top_volatility = df_imp.sort_values('volatility_driver', ascending=False).head(3)
    print("\n📈 TOP CEILING DRIVERS (Matters more for Explosions/Overs):")
    for _, row in top_volatility.iterrows():
        print(f"   - {row['feature']} (Diff: +{row['volatility_driver']:.4f})")

    top_stability = df_imp.sort_values('volatility_driver', ascending=True).head(3)
    print("\n🛡️ TOP FLOOR DRIVERS (Matters more for Duds/Unders):")
    for _, row in top_stability.iterrows():
        print(f"   - {row['feature']} (Diff: {row['volatility_driver']:.4f})")

if __name__ == "__main__":
    print("Loading feature names...")
    feats = get_feature_names()

    if not feats:
        print("Please ensure you have trained the models and generated one of: "+", ".join(PREFERRED_FEATURE_FILES))
        raise SystemExit(2)

    print(f"✅ Found {len(feats)} features.")

    model_files = discover_model_files()
    if not model_files:
        print("No quantile model files found. Expected pattern: 'nba_quantile_*.pkl' in data/.")
        raise SystemExit(3)

    print(f"Found {len(model_files)} quantile model files: {', '.join(model_files)}")
    for f in model_files:
        analyze_model(f, feats)

    print('\nDone.')
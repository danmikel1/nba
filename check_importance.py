import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model and feature list
try:
    model = joblib.load("data/nba_model_universal.pkl")
    features = joblib.load("data/nba_features_v20.pkl")

    # Get importance
    importance = model.feature_importances_
    
    # Create DataFrame
    df_imp = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print("\n🏆 TOP 10 MOST IMPORTANT FEATURES:")
    print(df_imp.head(10))
    
    # Check your new babies
    print("\n🆕 NEW FEATURE RANKINGS:")
    new_features = [
        'feat_usage_rate', 
        'feat_h2h_avg', 
        'feat_eff_per_min',   # <--- Added
        'feat_fatigue_load',  # <--- Added
        'feat_form_gap'       # <--- Added
    ]
    print(df_imp[df_imp['feature'].isin(new_features)])

except Exception as e:
    print(f"Error checking importance: {e}")
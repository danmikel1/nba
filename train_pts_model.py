import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

# 1. Configuration
CSV_FILE = 'ml_training_data_pts.csv'
MODEL_DIR = '.'
MODEL_FILE = 'nba_model.pkl'

print(f"🚀 Loading data from {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)

# 2. Filter & Clean
# Ensure we only train on valid rows (Hit = 1 or 0)
df = df.dropna(subset=['hit'])
df['hit'] = df['hit'].astype(int)

# 3. Select Features
# Automatically grab every column that starts with "feat_"
feature_cols = [c for c in df.columns if c.startswith('feat_')]
print(f"✅ Found {len(feature_cols)} features (Spread, Rest, Matchup, etc.)")

X = df[feature_cols]
y = df['hit']

# 4. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Brain (XGBoost)
print("🧠 Training XGBoost Model...")
model = xgb.XGBClassifier(
    n_estimators=100,      # Number of decision trees
    learning_rate=0.05,    # Learn slowly to avoid overfitting
    max_depth=3,           # Keep trees shallow (prevents memorization)
    eval_metric='logloss',
    use_label_encoder=False
)
model.fit(X_train, y_train)

# 6. Evaluate
preds = model.predict(X_test)
precision = precision_score(y_test, preds)
accuracy = accuracy_score(y_test, preds)

print("-" * 30)
print(f"🏆 MODEL RESULTS (Test Set)")
print(f"Precision: {precision:.1%} (Win Rate on Predictions)")
print(f"Accuracy:  {accuracy:.1%} (Overall Correctness)")
print("-" * 30)

# 7. Save the Brain
# Ensure the 'data' folder exists
os.makedirs(MODEL_DIR, exist_ok=True)
save_path = os.path.join(MODEL_DIR, MODEL_FILE)

joblib.dump(model, save_path)
print(f"💾 Model saved to: {save_path}")
print("✅ READY. Your app will now use this brain!")
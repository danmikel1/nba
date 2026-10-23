import pandas as pd
p='data/ml_training_data.csv'
print('Reading header...')
cols = pd.read_csv(p, nrows=0).columns.tolist()
feat_cols = [c for c in cols if c.startswith('feat_')]
print(f'Found {len(feat_cols)} feat_ columns; sampling correlations for all')
usecols = ['hit'] + feat_cols
print('Loading columns...')
df = pd.read_csv(p, usecols=usecols)
# Coerce to numeric where possible
for c in feat_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

cors = df[feat_cols].corrwith(df['hit']).abs().sort_values(ascending=False)
print('\nTop 20 feature |abs(corr with hit)|')
print(cors.head(20))

suspicious = cors[cors>0.5]
print('\nFeatures with abs(corr) > 0.5 (suspicious):')
print(suspicious)

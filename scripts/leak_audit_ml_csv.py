import pandas as pd
p='data/ml_training_data.csv'
print('Reading header...')
cols = pd.read_csv(p, nrows=0).columns.tolist()
feat_cols = [c for c in cols if c.startswith('feat_')]
meta_cols = [c for c in cols if not c.startswith('feat_')]
print(f'Feature columns ({len(feat_cols)}): sample ->', feat_cols[:10])
print('Metadata columns sample ->', meta_cols[:20])

# Sanity: ensure no feature column is actually a label or prediction metadata
for bad in ('actual_value','hit','result','predicted_prob','predicted_ev','projected_value','predicted_mean'):
    if bad in feat_cols:
        print('LEAK: feature column equals reserved col ->', bad)

# Load necessary columns for numeric checks
usecols = ['projected_value','actual_value','predicted_prob','hit'] + feat_cols
print('Loading', len(usecols), 'columns (may be large)...')
df = pd.read_csv(p, usecols=[c for c in usecols if c in cols])

# 1) projected_value == actual_value fraction
if 'projected_value' in df.columns and 'actual_value' in df.columns:
    same = (df['projected_value'] == df['actual_value']).mean()
    print('Fraction where projected_value == actual_value:', same)

# 2) feature equality to actual_value (exact match fraction)
match_counts = {}
for c in feat_cols:
    if c in df.columns:
        match_counts[c] = (df[c] == df['actual_value']).mean()
# show any with >0.01
susp = {k:v for k,v in match_counts.items() if v>0.01}
print('Feat columns exactly matching actual_value >1% rows:', susp)

# 3) Pearson correlation abs(feat, actual_value) top 10
num_feats = [c for c in feat_cols if c in df.columns]
corrs = df[num_feats].corrwith(df['actual_value']).abs().sort_values(ascending=False)
print('\nTop 10 abs(corr with actual_value):')
print(corrs.head(10))

# 4) Pearson correlation abs(feat, hit) top 10
corrs_hit = df[num_feats].corrwith(df['hit']).abs().sort_values(ascending=False)
print('\nTop 10 abs(corr with hit):')
print(corrs_hit.head(10))

# 5) Any feature perfectly correlated (>=0.99) with actual_value or hit?
pc = corrs[corrs>=0.99]
nph = corrs_hit[corrs_hit>=0.9]
print('\nFeatures with abs(corr)>=0.99 with actual_value:', list(pc.index))
print('Features with abs(corr)>=0.90 with hit:', list(nph.index))

# 6) Quick temporal check for suspicious projected_value distribution vs actual
if 'projected_value' in df.columns:
    print('\nProjected_value stats:')
    print(df['projected_value'].describe())
    print('\nActual_value stats:')
    print(df['actual_value'].describe())

print('\nDone.')

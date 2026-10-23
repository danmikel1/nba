import pandas as pd
p='data/ml_training_data.csv'
print('Reading header...')
cols = pd.read_csv(p, nrows=0).columns.tolist()
print('Columns present:', [c for c in ['date','feat_usage_rate','feat_min','feat_eff_per_min'] if c in cols])
usecols = ['date','feat_usage_rate']
if 'feat_min' in cols:
    usecols.append('feat_min')
if 'feat_eff_per_min' in cols:
    usecols.append('feat_eff_per_min')

df = pd.read_csv(p, usecols=usecols, parse_dates=['date'])
print('\nUsage rate stats:')
print(df['feat_usage_rate'].describe())
print('\nCount feat_usage_rate > 0.5:', (df['feat_usage_rate']>0.5).sum())
print('Count feat_usage_rate > 0.35:', (df['feat_usage_rate']>0.35).sum())
if 'feat_min' in df.columns:
    print('\nFound feat_min column (should be removed); sample values:')
    print(df['feat_min'].dropna().sample(min(10,len(df['feat_min'].dropna()))))
else:
    print('\nNo feat_min column present (good)')

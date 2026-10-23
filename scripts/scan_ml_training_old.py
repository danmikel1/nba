import pandas as pd
p = 'data/ml_training_data_old.csv'
print('Reading minimal columns from', p)
df = pd.read_csv(p, usecols=['date','hit','result'], parse_dates=['date'])
print('Rows:', len(df))
# overall
if 'hit' in df.columns:
    win_rate = df['hit'].mean()
else:
    win_rate = (df['result']=='Win').mean()
print(f'Overall win rate (hit): {win_rate:.4%}')
print('Date range:', df['date'].min(), 'to', df['date'].max())
print('\nWin rate by year:')
byyr = df.groupby(df['date'].dt.year)['hit'].agg(['count','mean']).sort_index(ascending=False)
print(byyr.head(10))
print('\nTop markets by win rate:')
# sample market breakdown if present
try:
    df2 = pd.read_csv(p, usecols=['market','hit'])
    print(df2.groupby('market')['hit'].agg(['count','mean']).sort_values('mean', ascending=False).head(20))
except Exception:
    pass

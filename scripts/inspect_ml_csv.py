import pandas as pd

p = 'data/ml_training_data.csv'
print('Loading minimal columns from', p)
df = pd.read_csv(p, usecols=['date', 'result', 'hit'], parse_dates=['date'])
print('Rows:', len(df))
print('Overall win rate (hit):', f"{df['hit'].mean():.4%}")
print('Date range:', df['date'].min(), 'to', df['date'].max())

print('\nWin rate by year:')
byyr = df.groupby(df['date'].dt.year)['hit'].agg(['count', 'mean']).sort_index(ascending=False)
print(byyr.head(10))

print('\nMost recent 10 dates:')
print(df.groupby('date')['hit'].agg(['count', 'mean']).sort_index(ascending=False).head(10))

import pandas as pd
p='data/ml_training_data_old.csv'
df=pd.read_csv(p,usecols=['player','hit'])
overall = df['hit'].mean()
by_player = df.groupby('player')['hit'].agg(['count','mean']).reset_index()
sus = by_player[(by_player['count']>=500)&(by_player['mean']>0.7)]['player'].tolist()
print('Suspect players:', sus)
filtered = df[~df['player'].isin(sus)]
print('Original overall win rate:', overall)
print('Overall excluding suspect players:', filtered['hit'].mean())
# Contribution by sample size buckets
by_bucket = by_player.copy()
by_bucket['bucket'] = pd.cut(by_bucket['count'], bins=[0,50,200,500,2000,10000], labels=['<=50','51-200','201-500','501-2000','2001+'])
merged = df.merge(by_bucket[['player','bucket']], on='player')
print('\nWin rate by player-sample-size bucket:')
print(merged.groupby('bucket')['hit'].agg(['count','mean']))

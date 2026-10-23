import pandas as pd
p='data/ml_training_data_old.csv'
df=pd.read_csv(p,usecols=['player','hit'])
by_player = df.groupby('player')['hit'].agg(['count','mean']).reset_index()
by_player = by_player.sort_values(['count','mean'], ascending=[False,False])
print('Top 20 players by sample count:')
print(by_player.head(20).to_string(index=False))
# players with >=500 samples and win_rate>0.7
high = by_player[(by_player['count']>=500)&(by_player['mean']>0.7)]
print('\nPlayers with >=500 samples and win_rate>70% (should be none):')
print(high.to_string(index=False))
# distribution
print('\nDistribution of per-player win rates (players with >=50 samples):')
small = by_player[by_player['count']>=50]['mean']
print('Median:', small.median(), '75th:', small.quantile(0.75), '90th:', small.quantile(0.9))

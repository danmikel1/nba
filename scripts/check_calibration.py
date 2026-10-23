import pandas as pd
p='data/ml_training_data.csv'
print('Loading predicted_prob + hit (sample)')
df = pd.read_csv(p, usecols=['predicted_prob','hit']).dropna()
print('Total rows:', len(df))
# Create buckets
bins = [0.5,0.55,0.60,0.65,0.70,0.75,1.0]
labels = ['50-55','55-60','60-65','65-70','70-75','75-100']
df['bucket'] = pd.cut(df['predicted_prob'], bins=bins, labels=labels, include_lowest=True)
cal = df.groupby('bucket').agg(count=('hit','size'), pred_mean=('predicted_prob','mean'), actual_mean=('hit','mean'))
print(cal)

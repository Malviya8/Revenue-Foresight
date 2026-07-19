import pandas as pd

df = pd.read_csv('output/predictions.csv')

print('=== AGGREGATE ===')
agg = df[df['level']=='aggregate'][['horizon_days','p10_revenue','p50_revenue','p90_revenue','p10_roas','p50_roas','p90_roas']]
print(agg.to_string())

print('\n=== ROAS CAP CHECK ===')
print('Rows with p50_roas >= 50:', (df['p50_roas'] >= 50).sum())
print('Rows with p90_roas == 100:', (df['p90_roas'] == 100).sum())

print('\n=== SPREAD CHECK ===')
df['spread_ratio'] = df['p90_revenue'] / df['p10_revenue'].replace(0, 0.01)
print('Mean spread ratio:', round(df['spread_ratio'].mean(), 2))
print('Max spread ratio:', round(df['spread_ratio'].max(), 2))

print('\n=== P10 ROAS FLOOR CHECK ===')
# Floor applies when median outlook is healthy (p50_roas >= 2).
healthy = df['p50_roas'] >= 2.0
print('Healthy rows with p10_roas < 1:', int(((healthy) & (df['p10_roas'] < 1)).sum()))
print('Unhealthy rows with p10_roas < 1 (p50_roas < 2):', int(((~healthy) & (df['p10_roas'] < 1)).sum()))
print('Min p10_roas:', round(df['p10_roas'].min(), 3))
print(
    'Min p10_roas (healthy only):',
    round(df.loc[healthy, 'p10_roas'].min(), 3) if healthy.any() else 'n/a',
)

import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
import joblib

scaler = StandardScaler()
train_files = glob.glob('train/data_train_*.csv')
train_dfs = [pd.read_csv(f) for f in train_files]
train_all = pd.concat(train_dfs)
scaler.fit(train_all[['day', 'temperature']])
joblib.dump(scaler, 'scaler.pkl')

for f in train_files:
    df = pd.read_csv(f)
    df[['day', 'temperature']] = scaler.transform(df[['day', 'temperature']])
    df.to_csv(f, index=False)

test_files = glob.glob('test/data_test_*.csv')
for f in test_files:
    df = pd.read_csv(f)
    df[['day', 'temperature']] = scaler.transform(df[['day', 'temperature']])
    df.to_csv(f, index=False)

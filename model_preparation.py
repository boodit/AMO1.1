import pandas as pd
import glob
from sklearn.linear_model import LinearRegression
import joblib

train_files = glob.glob('train/data_train_*.csv')
train_dfs = [pd.read_csv(f) for f in train_files]
train_all = pd.concat(train_dfs)

X = train_all[['day']]
y = train_all['temperature']

model = LinearRegression()
model.fit(X, y)
joblib.dump(model, 'model.pkl')

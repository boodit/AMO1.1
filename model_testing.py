import pandas as pd
import glob
import joblib
from sklearn.metrics import mean_absolute_error

model = joblib.load('model.pkl')
test_files = glob.glob('test/data_test_*.csv')
mae_list = []

for f in test_files:
    df = pd.read_csv(f)
    X = df[['day']]
    y_true = df['temperature']
    y_pred = model.predict(X)
    mae = mean_absolute_error(y_true, y_pred)
    print(f'{f}: MAE = {mae:.3f}')
    mae_list.append(mae)

print(f'Средняя MAE по всем тестовым наборам: {sum(mae_list)/len(mae_list):.3f}')

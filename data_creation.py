# data_creation.py
import numpy as np
import pandas as pd
import os

np.random.seed(42)

def create_dataset(size, noise_level=1.0, anomalies=False):
    days = np.arange(size)
    # Моделируем синусоидальные изменения температуры
    temp = 10 + 10 * np.sin(days * 2 * np.pi / 365) + np.random.normal(0, noise_level, size)
    if anomalies:
        idx = np.random.choice(size, size // 10)
        temp[idx] += np.random.normal(15, 5, len(idx))
    return pd.DataFrame({'day': days, 'temperature': temp})

os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)

# Создаем несколько наборов
for i in range(3):
    df = create_dataset(365, noise_level=1.0, anomalies=(i==2))
    df.to_csv(f'train/data_train_{i}.csv', index=False)

for i in range(2):
    df = create_dataset(100, noise_level=2.0, anomalies=(i==1))
    df.to_csv(f'test/data_test_{i}.csv', index=False)

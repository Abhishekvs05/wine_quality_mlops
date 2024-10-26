# src/data/data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import DATA_PATH, SEED


def load_and_preprocess_data(filepath=DATA_PATH):
    df = pd.read_csv(filepath, delimiter=';')
    X = df.drop('quality', axis=1)
    y = df['quality']

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data
    return train_test_split(X, y, test_size=0.2, random_state=SEED)

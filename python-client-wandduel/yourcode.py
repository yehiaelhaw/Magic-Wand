import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

spellname1 = "Scissors"
spellname2 = "Paper"
spellname3 = "Rock"

model = None

BASE_DIR = r"..\python-recorder\recordings"
LABEL_MAP = {
    "Scissors Upwards": 1,
    "Scissors Downwards": 1,
    "Paper left direction": 2,
    "Paper right direction": 2,
    "Rock Clockwise": 3,
    "Rock Anticlockwise": 3,
}

def load_data():
    data = []
    labels = []

    for folder_name, label in LABEL_MAP.items():
        folder_path = os.path.join(BASE_DIR, folder_name)
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path, sep=';')
                df = df.iloc[:, 3:-1]
                flattened = df.values.flatten()
                data.append(flattened)
                labels.append(label)

    return pd.DataFrame(data), pd.Series(labels)

def train_model():
    global model
    X, y = load_data()
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=12))
    ])
    pipeline.fit(X_train, y_train)
    model = pipeline

def process_spell(pandas_df: pd.DataFrame):
    global model
    if model is None:
        train_model()

    df = pandas_df.iloc[:, 3:-1]
    expected_rows = 26

    if df.shape[0] < expected_rows:
        padding = pd.DataFrame(0, index=range(expected_rows - df.shape[0]), columns=df.columns)
        df = pd.concat([df, padding], ignore_index=True)
    elif df.shape[0] > expected_rows:
        df = df.iloc[:expected_rows]
    features = df.values.flatten().reshape(1, -1)
    print(np.size(features))
    prediction = model.predict(features)
    return int(prediction[0]), get_spellname(int(prediction[0]))

def get_spellname(id):
    if id == 1:
        return spellname1
    if id == 2:
        return spellname2
    if id == 3:
        return spellname3
    return "Unknown Spell"
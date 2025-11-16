#!/usr/bin/env python3
# Pokémon G1 OU – Outcome predictor
# Versione SOLO Logistic Regression 

import random
from typing import List
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from toolbox import build_train_dataset, build_test_dataset


SEED = 42
random.seed(SEED); np.random.seed(SEED)

TRAIN_FILE = "train.jsonl"
TEST_FILE  = "test.jsonl"
OUT_FILE   = "submission_logreg.csv"


def get_model():
    return Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(
            C=2.0,
            max_iter=2000,
            n_jobs=-1,
            random_state=SEED
        ))
    ])


def main():
    
    X,y = build_train_dataset(TRAIN_FILE) 
    model = get_model()

    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []
    for tr,va in skf.split(X,y):
        model.fit(X.iloc[tr], y[tr])
        pred = model.predict(X.iloc[va])
        scores.append(accuracy_score(y[va], pred))

    print(f"[CV] LogReg 5-fold accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    
    model.fit(X,y)

    
    X_test, ids = build_test_dataset(TEST_FILE, train_cols=X.columns.tolist()) 
    preds = model.predict(X_test)

    pd.DataFrame({
        "battle_id": ids,
        "player_won": preds.astype(int)
    }).to_csv(OUT_FILE, index=False)

    print(f"[OK] Wrote submission to '{OUT_FILE}'")

if __name__ == "__main__":
    main()



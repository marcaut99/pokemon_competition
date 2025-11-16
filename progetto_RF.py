#!/usr/bin/env python3
# Pokémon G1 OU – Outcome predictor (timeline + statiche)
# Versione SOLO Random Forest

import random
from typing import List, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from toolbox import build_train_dataset, build_test_dataset 

SEED = 42
random.seed(SEED); np.random.seed(SEED)

TRAIN_FILE = "train.jsonl"
TEST_FILE  = "test.jsonl"
OUT_FILE   = "submission_rf.csv"


def get_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=SEED,
    )


def main():
    # Chiama la funzione importata da toolbox
    X, y = build_train_dataset(TRAIN_FILE)
    model = get_model()

    # Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []
    for tr, va in skf.split(X, y):
        model.fit(X.iloc[tr], y[tr])
        preds = model.predict(X.iloc[va])
        scores.append(accuracy_score(y[va], preds))
    print(f"[CV] RF 5-fold accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # Training finale e Submission
    model.fit(X, y)

    # Chiama la funzione importata da toolbox
    X_test, ids = build_test_dataset(TEST_FILE, train_cols=X.columns.tolist())
    preds = model.predict(X_test)

    pd.DataFrame({
        "battle_id": ids,
        "player_won": preds.astype(int)
    }).to_csv(OUT_FILE, index=False, encoding="utf-8")

    print(f"[OK] Wrote submission to '{OUT_FILE}' with shape ({len(ids)}, 2).")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# Pokémon G1 OU – Outcome predictor
# Modello principale: XGBoost
# Usa feature statiche (team P1 + lead P2) + feature dinamiche (timeline primi 30 turni)

import random
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier # XGBoost come modello principale

from toolbox import build_train_dataset, build_test_dataset

SEED = 42
random.seed(SEED); np.random.seed(SEED)

TRAIN_FILE = "train.jsonl"
TEST_FILE  = "test.jsonl"
OUT_FILE   = "submission_xgb.csv"


def get_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=900,
        max_depth=5,
        min_child_weight=3,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        reg_alpha=0.1,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED,
    )


def main():
    
    X, y = build_train_dataset(TRAIN_FILE)

    model = get_model()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []
    for tr, va in skf.split(X, y):
        model.fit(X.iloc[tr], y[tr])
        preds = model.predict(X.iloc[va])
        scores.append(accuracy_score(y[va], preds))
    print(f"[CV] XGB 5-fold accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    
    model.fit(X, y)

    
    X_test, ids = build_test_dataset(TEST_FILE, train_cols=X.columns.tolist())
    preds_test = model.predict(X_test)

    pd.DataFrame({
        "battle_id": ids,
        "player_won": preds_test.astype(int)
    }).to_csv(OUT_FILE, index=False, encoding="utf-8")

    print(f"[OK] Wrote submission to '{OUT_FILE}' with shape ({len(ids)}, 2).")


if __name__ == "__main__":
    main()





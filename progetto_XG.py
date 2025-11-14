#!/usr/bin/env python3
# Pokémon G1 OU – Outcome predictor (timeline-only, rules-compliant)
# Versione SOLO XG 

import json, re, random
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
from sklearn.ensemble import RandomForestClassifier

SEED = 42
random.seed(SEED); np.random.seed(SEED)

TRAIN_FILE = "train.jsonl"
TEST_FILE  = "test.jsonl"
OUT_FILE   = "submission.csv"

_HP_KEY_RE = re.compile(r"(hp).*?(pct|percent)", re.IGNORECASE)

def _safe_float(v: Any, default: float=0.0) -> float:
    try:
        x = float(v)
        if np.isnan(x): return default
        return x
    except Exception:
        return default

def _hp_pct_from_state(state: Dict[str, Any]) -> float:
    if not isinstance(state, dict): return 100.0
    for k, v in state.items():
        if isinstance(k, str) and _HP_KEY_RE.search(k):
            return max(0.0, min(100.0, _safe_float(v, 100.0)))
    if "hp" in state:
        val = _safe_float(state.get("hp"), 100.0)
        if 0.0 <= val <= 100.0:
            return max(0.0, min(100.0, val))
    return 100.0

def _status_str(state: Dict[str, Any]) -> str:
    if not isinstance(state, dict): return ""
    for key in ("status", "ailment", "condition"):
        v = state.get(key)
        if isinstance(v, str): return v.lower()
    return ""

def _get_timeline(battle: Dict[str, Any]) -> List[Dict[str, Any]]:
    tl = battle.get("battle_timeline")
    if isinstance(tl, list): return tl[:30]
    tl = battle.get("timeline")
    if isinstance(tl, list): return tl[:30]
    return []

def _has_move(md: Any) -> bool:
    if isinstance(md, dict): return True
    if isinstance(md, list): return len(md) > 0
    return False

def extract_timeline_features(battle: Dict[str, Any]) -> Dict[str, float]:
    if "battle_id" not in battle: raise KeyError("Missing 'battle_id'.")
    feats: Dict[str, float] = {"turns": 0.0}
    tl = _get_timeline(battle); n = len(tl); feats["turns"] = float(n)

    p1_hp=[]; p2_hp=[]; p1_moves=0; p2_moves=0
    p1_bp_sum=0.0; p2_bp_sum=0.0
    p1_status_inflicted=0; p2_status_inflicted=0
    p1_kos=0; p2_kos=0

    if n>0:
        first = tl[0]
        prev_p1_hp = _hp_pct_from_state(first.get("p1_pokemon_state") or first.get("player1") or {})
        prev_p2_hp = _hp_pct_from_state(first.get("p2_pokemon_state") or first.get("player2") or {})
    else:
        prev_p1_hp = prev_p2_hp = 100.0

    first15_diffs=[]; first30_diffs=[]

    for i, ev in enumerate(tl):
        s1 = ev.get("p1_pokemon_state") or ev.get("player1") or {}
        s2 = ev.get("p2_pokemon_state") or ev.get("player2") or {}
        hp1 = _hp_pct_from_state(s1); hp2 = _hp_pct_from_state(s2)
        p1_hp.append(hp1); p2_hp.append(hp2)

        md1 = ev.get("p1_move_details") or ev.get("p1_action") or ev.get("action_p1")
        md2 = ev.get("p2_move_details") or ev.get("p2_action") or ev.get("action_p2")
        if _has_move(md1):
            p1_moves += 1
            if isinstance(md1, dict): p1_bp_sum += _safe_float(md1.get("base_power", 0.0), 0.0)
        if _has_move(md2):
            p2_moves += 1
            if isinstance(md2, dict): p2_bp_sum += _safe_float(md2.get("base_power", 0.0), 0.0)

        st2 = _status_str(s2); st1 = _status_str(s1)
        if st2 not in ("", "none", "nostatus"): p1_status_inflicted += 1
        if st1 not in ("", "none", "nostatus"): p2_status_inflicted += 1

        if hp2 <= 0.0 and prev_p2_hp > 0.0: p1_kos += 1
        if hp1 <= 0.0 and prev_p1_hp > 0.0: p2_kos += 1
        prev_p1_hp, prev_p2_hp = hp1, hp2

        if i < 15: first15_diffs.append(hp1 - hp2)
        if i < 30: first30_diffs.append(hp1 - hp2)

    def _stats(arr: List[float]) -> Tuple[float, float, float, float]:
        if not arr: return (100.0, 0.0, 100.0, 100.0)
        return (float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr)))

    p1_mean,p1_std,p1_min,_ = _stats(p1_hp)
    p2_mean,p2_std,p2_min,_ = _stats(p2_hp)

    def _diffsum(arr: List[float]) -> Tuple[float, float]:
        if len(arr) <= 1: return (0.0, 0.0)
        diffs = [arr[i] - arr[i-1] for i in range(1, len(arr))]
        dmg = -sum(d for d in diffs if d < 0); rec = sum(d for d in diffs if d > 0)
        return float(dmg), float(rec)

    p1_dmg,p1_rec = _diffsum(p1_hp); p2_dmg,p2_rec = _diffsum(p2_hp)

    feats.update({
        "p1_hp_mean": p1_mean, "p1_hp_std": p1_std, "p1_hp_min": p1_min,
        "p2_hp_mean": p2_mean, "p2_hp_std": p2_std, "p2_hp_min": p2_min,
        "p1_hp_last": (p1_hp[-1] if p1_hp else 100.0),
        "p2_hp_last": (p2_hp[-1] if p2_hp else 100.0),
        "p1_moves": float(p1_moves), "p2_moves": float(p2_moves),
        "p1_bp_mean": (p1_bp_sum/p1_moves) if p1_moves>0 else 0.0,
        "p2_bp_mean": (p2_bp_sum/p2_moves) if p2_moves>0 else 0.0,
        "p1_status_inflicted": float(p1_status_inflicted), "p2_status_inflicted": float(p2_status_inflicted),
        "p1_kos": float(p1_kos), "p2_kos": float(p2_kos),
        "p1_dmg_sum": p1_dmg, "p1_rec_sum": p1_rec, "p2_dmg_sum": p2_dmg, "p2_rec_sum": p2_rec,
        "rel_first15_hp": float(np.mean(first15_diffs)) if first15_diffs else 0.0,
        "rel_first30_hp": float(np.mean(first30_diffs)) if first30_diffs else 0.0,
    })

    def rel(a, b, name): feats[name] = feats.get(a, 0.0) - feats.get(b, 0.0)
    rel("p1_hp_mean","p2_hp_mean","rel_hp_mean")
    rel("p1_hp_std","p2_hp_std","rel_hp_std")
    rel("p1_hp_min","p2_hp_min","rel_hp_min")
    rel("p1_hp_last","p2_hp_last","rel_hp_last")
    rel("p1_moves","p2_moves","rel_moves")
    rel("p1_bp_mean","p2_bp_mean","rel_bp_mean")
    rel("p1_status_inflicted","p2_status_inflicted","rel_status_inflicted")
    rel("p1_kos","p2_kos","rel_kos")
    rel("p1_dmg_sum","p2_dmg_sum","rel_dmg_sum")
    rel("p1_rec_sum","p2_rec_sum","rel_rec_sum")

    total_moves = feats["p1_moves"] + feats["p2_moves"]
    move_ratio = (feats["p1_moves"]/total_moves) if total_moves>0 else 0.5
    feats["momentum_index"] = feats["rel_first15_hp"] * (1.0 + 0.5*(move_ratio - 0.5))
    return feats

def _load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: yield json.loads(line)

def build_train_dataset(path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    feats=[]; labels=[]
    for b in _load_jsonl(path):
        feats.append(extract_timeline_features(b))
        lbl = b.get("player_won")
        if isinstance(lbl, bool): labels.append(1 if lbl else 0)
        elif isinstance(lbl, (int, np.integer)): labels.append(int(lbl))
        elif isinstance(lbl, str): labels.append(1 if lbl.strip().lower()=="true" else 0)
        else: raise ValueError("Unrecognized label.")
    X = pd.DataFrame(feats).fillna(0.0)
    nunique = X.nunique()
    keep = nunique[nunique>1].index.tolist()
    X = X[keep]
    y = np.array(labels, dtype=int)
    return X, y

def build_test_dataset(path: str, train_cols: List[str]) -> Tuple[pd.DataFrame, List[Any]]:
    feats=[]; ids=[]
    for b in _load_jsonl(path):
        if "battle_id" not in b: raise KeyError("Missing 'battle_id' in test.")
        feats.append(extract_timeline_features(b))
        ids.append(b["battle_id"])
    X = pd.DataFrame(feats).fillna(0.0)
    for c in train_cols:
        if c not in X.columns: X[c] = 0.0
    X = X[train_cols]
    return X, ids

def get_model():
    if HAS_XGB:
        return XGBClassifier(
            n_estimators=800, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, random_state=SEED, n_jobs=-1, tree_method="hist"
        )
    return RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1,
        random_state=SEED, n_jobs=-1
    )

def main():
    X, y = build_train_dataset(TRAIN_FILE)
    model = get_model()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []
    for tr, va in skf.split(X, y):
        model.fit(X.iloc[tr], y[tr])
        scores.append(accuracy_score(y[va], model.predict(X.iloc[va])))
    print(f"[CV] {'XGB' if HAS_XGB else 'RF'} 5-fold accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    model.fit(X, y)
    X_test, ids = build_test_dataset(TEST_FILE, train_cols=X.columns.tolist())
    preds = model.predict(X_test)
    pd.DataFrame({
        "battle_id": ids,
        "player_won": preds.astype(int)
    }).to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"[OK] Wrote submission to '{OUT_FILE}' with shape ({len(ids)}, 2).")

if __name__ == "__main__":
    main()
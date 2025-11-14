#!/usr/bin/env python3
# Pokémon G1 OU – Outcome predictor
# Modello principale: XGBoost
# Usa feature statiche (team P1 + lead P2) + feature dinamiche (timeline primi 30 turni)

import json, re, random
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier  # XGBoost come modello principale

SEED = 42
random.seed(SEED); np.random.seed(SEED)

TRAIN_FILE = "train.jsonl"
TEST_FILE  = "test.jsonl"
OUT_FILE   = "submission_xgb.csv"

_HP_KEY_RE = re.compile(r"(hp).*?(pct|percent)", re.IGNORECASE)


# ----------------------- Helper di base ----------------------- #

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if np.isnan(x):
            return default
        return x
    except Exception:
        return default


def _hp_pct_from_state(state: Dict[str, Any]) -> float:
    if not isinstance(state, dict):
        return 100.0
    for k, v in state.items():
        if isinstance(k, str) and _HP_KEY_RE.search(k):
            return max(0.0, min(100.0, _safe_float(v, 100.0)))
    if "hp" in state:
        val = _safe_float(state.get("hp"), 100.0)
        if 0.0 <= val <= 100.0:
            return max(0.0, min(100.0, val))
    return 100.0


def _status_str(state: Dict[str, Any]) -> str:
    if not isinstance(state, dict):
        return ""
    for key in ("status", "ailment", "condition"):
        v = state.get(key)
        if isinstance(v, str):
            return v.lower()
    return ""


def _get_timeline(battle: Dict[str, Any]) -> List[Dict[str, Any]]:
    tl = battle.get("battle_timeline")
    if isinstance(tl, list):
        return tl[:30]
    tl = battle.get("timeline")
    if isinstance(tl, list):
        return tl[:30]
    return []


def _has_move(md: Any) -> bool:
    if isinstance(md, dict):
        return True
    if isinstance(md, list):
        return len(md) > 0
    return False


# ----------------------- Feature Engineering ----------------------- #

def extract_features(battle: Dict[str, Any]) -> Dict[str, float]:
    if "battle_id" not in battle:
        raise KeyError("Missing 'battle_id'.")

    feats: Dict[str, float] = {"turns": 0.0}

    
    p1_team = battle.get("p1_team_details") or []
    p2_lead = battle.get("p2_lead_details") or {}

    if isinstance(p1_team, list) and len(p1_team) > 0:
        def _team_mean(key: str) -> float:
            vals = [p.get(key, 0.0) for p in p1_team if isinstance(p, dict)]
            return float(np.mean(vals)) if vals else 0.0

        def _team_sum(key: str) -> float:
            vals = [p.get(key, 0.0) for p in p1_team if isinstance(p, dict)]
            return float(np.sum(vals)) if vals else 0.0

        feats["p1_team_mean_hp"]  = _team_mean("base_hp")
        feats["p1_team_mean_atk"] = _team_mean("base_atk")
        feats["p1_team_mean_def"] = _team_mean("base_def")
        feats["p1_team_mean_spa"] = _team_mean("base_spa")
        feats["p1_team_mean_spd"] = _team_mean("base_spd")
        feats["p1_team_mean_spe"] = _team_mean("base_spe")
        feats["p1_team_sum_hp"]   = _team_sum("base_hp")
    else:
        feats["p1_team_mean_hp"]  = 0.0
        feats["p1_team_mean_atk"] = 0.0
        feats["p1_team_mean_def"] = 0.0
        feats["p1_team_mean_spa"] = 0.0
        feats["p1_team_mean_spd"] = 0.0
        feats["p1_team_mean_spe"] = 0.0
        feats["p1_team_sum_hp"]   = 0.0

    if isinstance(p2_lead, dict) and len(p2_lead) > 0:
        feats["p2_lead_hp"]  = float(p2_lead.get("base_hp", 0.0))
        feats["p2_lead_atk"] = float(p2_lead.get("base_atk", 0.0))
        feats["p2_lead_def"] = float(p2_lead.get("base_def", 0.0))
        feats["p2_lead_spa"] = float(p2_lead.get("base_spa", 0.0))
        feats["p2_lead_spd"] = float(p2_lead.get("base_spd", 0.0))
        feats["p2_lead_spe"] = float(p2_lead.get("base_spe", 0.0))
    else:
        feats["p2_lead_hp"]  = 0.0
        feats["p2_lead_atk"] = 0.0
        feats["p2_lead_def"] = 0.0
        feats["p2_lead_spa"] = 0.0
        feats["p2_lead_spd"] = 0.0
        feats["p2_lead_spe"] = 0.0

    feats["rel_static_hp"]  = feats["p1_team_mean_hp"]  - feats["p2_lead_hp"]
    feats["rel_static_atk"] = feats["p1_team_mean_atk"] - feats["p2_lead_atk"]
    feats["rel_static_def"] = feats["p1_team_mean_def"] - feats["p2_lead_def"]
    feats["rel_static_spa"] = feats["p1_team_mean_spa"] - feats["p2_lead_spa"]
    feats["rel_static_spd"] = feats["p1_team_mean_spd"] - feats["p2_lead_spd"]
    feats["rel_static_spe"] = feats["p1_team_mean_spe"] - feats["p2_lead_spe"]

    
    tl = _get_timeline(battle)
    n = len(tl)
    feats["turns"] = float(n)

    p1_hp = []; p2_hp = []
    p1_moves = 0; p2_moves = 0
    p1_bp_sum = 0.0; p2_bp_sum = 0.0
    p1_status_inflicted = 0; p2_status_inflicted = 0
    p1_kos = 0; p2_kos = 0
    p1_first_ko_turn = 0
    p2_first_ko_turn = 0

    if n > 0:
        first = tl[0]
        prev_p1_hp = _hp_pct_from_state(first.get("p1_pokemon_state") or first.get("player1") or {})
        prev_p2_hp = _hp_pct_from_state(first.get("p2_pokemon_state") or first.get("player2") or {})
    else:
        prev_p1_hp = prev_p2_hp = 100.0

    
    hp_diffs_all = []
    hp_diffs_first8 = []

    
    early_indices = []   # turni 1-10
    mid_indices   = []   # turni 11-20
    late_indices  = []   # turni 21-30

    turns_p1_lead = 0
    turns_p2_lead = 0
    LEAD_THRESHOLD = 5.0  

    for i, ev in enumerate(tl):
        turn_idx = i + 1
        s1 = ev.get("p1_pokemon_state") or ev.get("player1") or {}
        s2 = ev.get("p2_pokemon_state") or ev.get("player2") or {}
        hp1 = _hp_pct_from_state(s1)
        hp2 = _hp_pct_from_state(s2)
        p1_hp.append(hp1); p2_hp.append(hp2)

        diff = hp1 - hp2
        hp_diffs_all.append(diff)
        if turn_idx <= 8:
            hp_diffs_first8.append(diff)

        if turn_idx <= 10:
            early_indices.append(i)
        elif turn_idx <= 20:
            mid_indices.append(i)
        else:
            late_indices.append(i)

        if diff > LEAD_THRESHOLD:
            turns_p1_lead += 1
        elif diff < -LEAD_THRESHOLD:
            turns_p2_lead += 1

        md1 = ev.get("p1_move_details") or ev.get("p1_action") or ev.get("action_p1")
        md2 = ev.get("p2_move_details") or ev.get("p2_action") or ev.get("action_p2")
        if _has_move(md1):
            p1_moves += 1
            if isinstance(md1, dict):
                p1_bp_sum += _safe_float(md1.get("base_power", 0.0), 0.0)
        if _has_move(md2):
            p2_moves += 1
            if isinstance(md2, dict):
                p2_bp_sum += _safe_float(md2.get("base_power", 0.0), 0.0)

        st2 = _status_str(s2); st1 = _status_str(s1)
        if st2 not in ("", "none", "nostatus"):
            p1_status_inflicted += 1
        if st1 not in ("", "none", "nostatus"):
            p2_status_inflicted += 1

        if hp2 <= 0.0 and prev_p2_hp > 0.0:
            p1_kos += 1
            if p1_first_ko_turn == 0:
                p1_first_ko_turn = turn_idx
        if hp1 <= 0.0 and prev_p1_hp > 0.0:
            p2_kos += 1
            if p2_first_ko_turn == 0:
                p2_first_ko_turn = turn_idx

        prev_p1_hp, prev_p2_hp = hp1, hp2

    def _stats(arr: List[float]) -> Tuple[float, float, float, float]:
        if not arr:
            return (100.0, 0.0, 100.0, 100.0)
        return float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))

    def _diffsum(arr: List[float]) -> Tuple[float, float]:
        if len(arr) <= 1:
            return (0.0, 0.0)
        diffs = [arr[i] - arr[i-1] for i in range(1, len(arr))]
        dmg = -sum(d for d in diffs if d < 0)
        rec =  sum(d for d in diffs if d > 0)
        return float(dmg), float(rec)

    p1_mean, p1_std, p1_min, _ = _stats(p1_hp)
    p2_mean, p2_std, p2_min, _ = _stats(p2_hp)
    p1_dmg, p1_rec = _diffsum(p1_hp)
    p2_dmg, p2_rec = _diffsum(p2_hp)

    feats.update({
        "p1_hp_mean": p1_mean, "p1_hp_std": p1_std, "p1_hp_min": p1_min,
        "p2_hp_mean": p2_mean, "p2_hp_std": p2_std, "p2_hp_min": p2_min,
        "p1_hp_last": (p1_hp[-1] if p1_hp else 100.0),
        "p2_hp_last": (p2_hp[-1] if p2_hp else 100.0),
        "p1_moves": float(p1_moves), "p2_moves": float(p2_moves),
        "p1_bp_mean": (p1_bp_sum / p1_moves) if p1_moves > 0 else 0.0,
        "p2_bp_mean": (p2_bp_sum / p2_moves) if p2_moves > 0 else 0.0,
        "p1_status_inflicted": float(p1_status_inflicted), "p2_status_inflicted": float(p2_status_inflicted),
        "p1_kos": float(p1_kos), "p2_kos": float(p2_kos),
        "p1_dmg_sum": p1_dmg, "p1_rec_sum": p1_rec,
        "p2_dmg_sum": p2_dmg, "p2_rec_sum": p2_rec,
    })

    
    feats["turns_p1_lead"] = float(turns_p1_lead)
    feats["turns_p2_lead"] = float(turns_p2_lead)
    feats["rel_turns_lead"] = float(turns_p1_lead - turns_p2_lead)

    
    feats["p1_first_ko_turn"] = float(p1_first_ko_turn)
    feats["p2_first_ko_turn"] = float(p2_first_ko_turn)
    feats["p1_first_blood"] = 1.0 if (
        p1_first_ko_turn > 0 and (p2_first_ko_turn == 0 or p1_first_ko_turn < p2_first_ko_turn)
    ) else 0.0
    feats["rel_first_ko_turn"] = float(
        (p2_first_ko_turn if p2_first_ko_turn > 0 else 40)
        - (p1_first_ko_turn if p1_first_ko_turn > 0 else 40)
    )

    
    def rel(a, b, name):
        feats[name] = feats.get(a, 0.0) - feats.get(b, 0.0)

    rel("p1_hp_mean", "p2_hp_mean", "rel_hp_mean")
    rel("p1_hp_std", "p2_hp_std", "rel_hp_std")
    rel("p1_hp_min", "p2_hp_min", "rel_hp_min")
    rel("p1_hp_last", "p2_hp_last", "rel_hp_last")
    rel("p1_moves", "p2_moves", "rel_moves")
    rel("p1_bp_mean", "p2_bp_mean", "rel_bp_mean")
    rel("p1_status_inflicted", "p2_status_inflicted", "rel_status_inflicted")
    rel("p1_kos", "p2_kos", "rel_kos")
    rel("p1_dmg_sum", "p2_dmg_sum", "rel_dmg_sum")
    rel("p1_rec_sum", "p2_rec_sum", "rel_rec_sum")

    total_moves = feats["p1_moves"] + feats["p2_moves"]
    move_ratio = (feats["p1_moves"] / total_moves) if total_moves > 0 else 0.5
    feats["momentum_index"] = feats["rel_hp_mean"] * (1.0 + 0.5 * (move_ratio - 0.5))

    
    def _segment_stats(indices: List[int], name_prefix: str):
        if not indices:
            feats[f"{name_prefix}_hp_diff_mean"] = 0.0
            feats[f"{name_prefix}_hp_diff_std"]  = 0.0
            feats[f"{name_prefix}_p1_lead_turns"] = 0.0
            return
        seg_diffs = [hp_diffs_all[i] for i in indices]
        feats[f"{name_prefix}_hp_diff_mean"] = float(np.mean(seg_diffs))
        feats[f"{name_prefix}_hp_diff_std"]  = float(np.std(seg_diffs))
        feats[f"{name_prefix}_p1_lead_turns"] = float(
            sum(1 for d in seg_diffs if d > LEAD_THRESHOLD)
        )

    _segment_stats(early_indices, "early")
    _segment_stats(mid_indices,   "mid")
    _segment_stats(late_indices,  "late")

    
    for t in range(8):
        if t < len(hp_diffs_first8):
            feats[f"hp_diff_t{t+1}"] = float(hp_diffs_first8[t])
        else:
            feats[f"hp_diff_t{t+1}"] = 0.0

    return feats



def _load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_train_dataset(path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    feats = []; labels = []
    for b in _load_jsonl(path):
        feats.append(extract_features(b))
        lbl = b.get("player_won")
        if isinstance(lbl, bool):
            labels.append(1 if lbl else 0)
        elif isinstance(lbl, (int, np.integer)):
            labels.append(int(lbl))
        elif isinstance(lbl, str):
            labels.append(1 if lbl.strip().lower() == "true" else 0)
        else:
            raise ValueError("Unrecognized label.")
    X = pd.DataFrame(feats).fillna(0.0)
    nunique = X.nunique()
    keep = nunique[nunique > 1].index.tolist()
    X = X[keep]
    y = np.array(labels, dtype=int)
    return X, y


def build_test_dataset(path: str, train_cols: List[str]) -> Tuple[pd.DataFrame, List[Any]]:
    feats = []; ids = []
    for b in _load_jsonl(path):
        if "battle_id" not in b:
            raise KeyError("Missing 'battle_id' in test.")
        feats.append(extract_features(b))
        ids.append(b["battle_id"])
    X = pd.DataFrame(feats).fillna(0.0)
    for c in train_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[train_cols]
    return X, ids


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

    # retrain su tutto il train
    model.fit(X, y)

    # test + submission
    X_test, ids = build_test_dataset(TEST_FILE, train_cols=X.columns.tolist())
    preds_test = model.predict(X_test)

    pd.DataFrame({
        "battle_id": ids,
        "player_won": preds_test.astype(int)
    }).to_csv(OUT_FILE, index=False, encoding="utf-8")

    print(f"[OK] Wrote submission to '{OUT_FILE}' with shape ({len(ids)}, 2).")


if __name__ == "__main__":
    main()




#!/usr/bin/env python3
# Pokémon G1 OU – Outcome predictor
# Versione SOLO Logistic Regression 

import json, re, random
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

SEED = 42
random.seed(SEED); np.random.seed(SEED)

TRAIN_FILE = "train.jsonl"
TEST_FILE  = "test.jsonl"
OUT_FILE   = "submission_logreg.csv"

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

def _get_timeline(battle: Dict[str, Any]):
    tl = battle.get("battle_timeline")
    if isinstance(tl, list): return tl[:30]
    tl = battle.get("timeline")
    if isinstance(tl, list): return tl[:30]
    return []

def _has_move(md: Any) -> bool:
    if isinstance(md, dict): return True
    if isinstance(md, list): return len(md) > 0
    return False



def extract_features(battle: Dict[str, Any]) -> Dict[str, float]:
    if "battle_id" not in battle:
        raise KeyError("Missing 'battle_id'.")

    feats: Dict[str, float] = {"turns": 0.0}

    
    p1_team = battle.get("p1_team_details") or []
    p2_lead = battle.get("p2_lead_details") or {}

    if isinstance(p1_team, list) and len(p1_team) > 0:

        def _team_mean(key): return float(np.mean([p.get(key,0.0) for p in p1_team]))
        def _team_sum(key):  return float(np.sum([p.get(key,0.0) for p in p1_team]))

        feats["p1_team_mean_hp"]  = _team_mean("base_hp")
        feats["p1_team_mean_atk"] = _team_mean("base_atk")
        feats["p1_team_mean_def"] = _team_mean("base_def")
        feats["p1_team_mean_spa"] = _team_mean("base_spa")
        feats["p1_team_mean_spd"] = _team_mean("base_spd")
        feats["p1_team_mean_spe"] = _team_mean("base_spe")
        feats["p1_team_sum_hp"]   = _team_sum("base_hp")
    else:
        feats["p1_team_mean_hp"] = feats["p1_team_mean_atk"] = feats["p1_team_mean_def"] = 0.0
        feats["p1_team_mean_spa"] = feats["p1_team_mean_spd"] = feats["p1_team_mean_spe"] = 0.0
        feats["p1_team_sum_hp"] = 0.0

    if isinstance(p2_lead, dict) and len(p2_lead) > 0:
        for k in ["base_hp","base_atk","base_def","base_spa","base_spd","base_spe"]:
            feats["p2_lead_"+k[5:]] = float(p2_lead.get(k,0.0))
    else:
        for v in ["hp","atk","def","spa","spd","spe"]:
            feats["p2_lead_"+v] = 0.0

    feats["rel_static_hp"]  = feats["p1_team_mean_hp"] - feats["p2_lead_hp"]
    feats["rel_static_atk"] = feats["p1_team_mean_atk"] - feats["p2_lead_atk"]
    feats["rel_static_def"] = feats["p1_team_mean_def"] - feats["p2_lead_def"]
    feats["rel_static_spa"] = feats["p1_team_mean_spa"] - feats["p2_lead_spa"]
    feats["rel_static_spd"] = feats["p1_team_mean_spd"] - feats["p2_lead_spd"]
    feats["rel_static_spe"] = feats["p1_team_mean_spe"] - feats["p2_lead_spe"]

    
    tl = _get_timeline(battle)
    n = len(tl)
    feats["turns"] = float(n)

    p1_hp=[]; p2_hp=[]
    hp_diffs_all=[]
    hp_diffs_first8=[]

    early=[]; mid=[]; late=[]

    turns_p1_lead=0; turns_p2_lead=0
    LEAD=5.0

    p1_moves=p2_moves=p1_kos=p2_kos=0
    p1_status_inflicted=p2_status_inflicted=0
    p1_bp=0.0; p2_bp=0.0
    p1_first_ko=0; p2_first_ko=0

    if n>0:
        first=tl[0]
        prev_p1=_hp_pct_from_state(first.get("p1_pokemon_state") or {})
        prev_p2=_hp_pct_from_state(first.get("p2_pokemon_state") or {})
    else:
        prev_p1=prev_p2=100.0

    for i,ev in enumerate(tl):
        t=i+1
        s1=ev.get("p1_pokemon_state") or {}
        s2=ev.get("p2_pokemon_state") or {}

        hp1=_hp_pct_from_state(s1)
        hp2=_hp_pct_from_state(s2)

        p1_hp.append(hp1)
        p2_hp.append(hp2)

        diff=hp1-hp2
        hp_diffs_all.append(diff)

        if t<=8: hp_diffs_first8.append(diff)
        if t<=10: early.append(i)
        elif t<=20: mid.append(i)
        else: late.append(i)

        if diff>LEAD: turns_p1_lead+=1
        elif diff<-LEAD: turns_p2_lead+=1

        md1=ev.get("p1_move_details")
        md2=ev.get("p2_move_details")

        if _has_move(md1):
            p1_moves+=1
            if isinstance(md1,dict): p1_bp+=_safe_float(md1.get("base_power",0.0))
        if _has_move(md2):
            p2_moves+=1
            if isinstance(md2,dict): p2_bp+=_safe_float(md2.get("base_power",0.0))

        st1=_status_str(s1)
        st2=_status_str(s2)

        if st2 not in ("","none","nostatus"): p1_status_inflicted+=1
        if st1 not in ("","none","nostatus"): p2_status_inflicted+=1

        if hp2<=0 and prev_p2>0:
            p1_kos+=1
            if p1_first_ko==0: p1_first_ko=t

        if hp1<=0 and prev_p1>0:
            p2_kos+=1
            if p2_first_ko==0: p2_first_ko=t

        prev_p1,prev_p2=hp1,hp2

    
    def _stats(a):
        if len(a)==0: return (100.0,0.0,100.0,100.0)
        return (float(np.mean(a)),float(np.std(a)),float(np.min(a)),float(np.max(a)))

    def _diffsum(a):
        if len(a)<=1: return (0.0,0.0)
        diffs=[a[i]-a[i-1] for i in range(1,len(a))]
        dmg=-sum(d for d in diffs if d<0)
        rec=sum(d for d in diffs if d>0)
        return float(dmg),float(rec)

    p1_m,p1_s,p1_min,_ = _stats(p1_hp)
    p2_m,p2_s,p2_min,_ = _stats(p2_hp)
    p1_d,p1_r = _diffsum(p1_hp)
    p2_d,p2_r = _diffsum(p2_hp)

    feats.update({
        "p1_hp_mean":p1_m,"p1_hp_std":p1_s,"p1_hp_min":p1_min,
        "p2_hp_mean":p2_m,"p2_hp_std":p2_s,"p2_hp_min":p2_min,
        "p1_hp_last":p1_hp[-1] if p1_hp else 100.0,
        "p2_hp_last":p2_hp[-1] if p2_hp else 100.0,
        "p1_moves":float(p1_moves),"p2_moves":float(p2_moves),
        "p1_bp_mean":p1_bp/max(p1_moves,1),
        "p2_bp_mean":p2_bp/max(p2_moves,1),
        "p1_status_inflicted":float(p1_status_inflicted),
        "p2_status_inflicted":float(p2_status_inflicted),
        "p1_kos":float(p1_kos),"p2_kos":float(p2_kos),
        "p1_dmg_sum":p1_d,"p1_rec_sum":p1_r,
        "p2_dmg_sum":p2_d,"p2_rec_sum":p2_r,
    })

    
    def rel(a,b,n): feats[n]=feats.get(a,0.0)-feats.get(b,0.0)

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

    feats["turns_p1_lead"]=float(turns_p1_lead)
    feats["turns_p2_lead"]=float(turns_p2_lead)
    feats["rel_turns_lead"]=float(turns_p1_lead - turns_p2_lead)

    feats["p1_first_ko_turn"]=float(p1_first_ko)
    feats["p2_first_ko_turn"]=float(p2_first_ko)
    feats["p1_first_blood"]=1.0 if (p1_first_ko>0 and (p2_first_ko==0 or p1_first_ko<p2_first_ko)) else 0.0
    feats["rel_first_ko_turn"]=float((p2_first_ko if p2_first_ko>0 else 40)-(p1_first_ko if p1_first_ko>0 else 40))

    
    def seg(idx,name):
        if not idx:
            feats[f"{name}_hp_diff_mean"]=0.0
            feats[f"{name}_hp_diff_std"]=0.0
            feats[f"{name}_p1_lead_turns"]=0.0
            return
        d=[hp_diffs_all[i] for i in idx]
        feats[f"{name}_hp_diff_mean"]=float(np.mean(d))
        feats[f"{name}_hp_diff_std"]=float(np.std(d))
        feats[f"{name}_p1_lead_turns"]=float(sum(1 for x in d if x>LEAD))

    seg(early,"early")
    seg(mid,"mid")
    seg(late,"late")

    
    for t in range(8):
        feats[f"hp_diff_t{t+1}"] = float(hp_diffs_first8[t]) if t<len(hp_diffs_first8) else 0.0

    return feats



def _load_jsonl(path: str):
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def build_train_dataset(path: str):
    feats=[]; y=[]
    for battle in _load_jsonl(path):
        feats.append(extract_features(battle))
        lbl = battle["player_won"]
        y.append(1 if lbl else 0)

    X=pd.DataFrame(feats).fillna(0.0)

    
    keep = X.columns[X.nunique()>1]
    X = X[keep]

    return X, np.array(y,dtype=int)

def build_test_dataset(path: str, train_cols: List[str]):
    feats=[]; ids=[]
    for battle in _load_jsonl(path):
        feats.append(extract_features(battle))
        ids.append(battle["battle_id"])

    X=pd.DataFrame(feats).fillna(0.0)
    
    for c in train_cols:
        if c not in X.columns: X[c]=0.0
    X=X[train_cols]
    return X,ids



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

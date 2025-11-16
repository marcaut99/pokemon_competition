# Pokémon G1 OU – Battle Outcome Prediction  
**Master's Degree in Data Science – Sapienza University of Rome**

This repository contains the complete solution developed for the Pokémon Generation 1 OU battle prediction challenge.

We built **three independent models** — Logistic Regression, Random Forest, and XGBoost — all using the **same feature-engineering pipeline** based on the official competition rules.  
Our final and best-performing model is **XGBoost**, with a public leaderboard score of **0.8173**.

---




---

## 🔍 Feature Engineering
All three models rely on the same feature extraction pipeline, which is centralized in the toolbox.py module to guarantee consistency and reproducibility across all scripts.

The features include:

Static features: Mean and total base stats of Player 1’s team, opponent lead Pokémon stats, and relative stat differences (P1 – P2).

Dynamic timeline features (first 30 turns): HP trajectories and per-turn HP differences, cumulative damage and recovery, First KO timing, KO counts, early/mid/late game advantage windows, move usage and mean base power, status conditions inflicted, and per-turn HP differential for turns 1–8.

---

## 🧪 Models

### **1. Logistic Regression**
A linear baseline model tuned over several values of `C`.  
Public score: ~0.809

### **2. Random Forest**
Tree-based model tested with different depths, estimators, and split criteria.  
Public score: ~0.8146

### **3. XGBoost (Final Model)**
Tuned with:
- learning rate  
- depth  
- subsampling  
- column sampling  
- regularization

Final Public Leaderboard Score: **0.8173**

---

## 📝 Notebook
The included `notebook.ipynb`:

1. Clones this repository  
2. Loads train.jsonl and test.jsonl  
3. Runs all three scripts  
4. Produces the three submission files  

---

## 📬 Authors
- **Gabriel Gaitanaru**  
- **Tommaso Montuori**  
- **Marco Autieri**

---



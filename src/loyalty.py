"""What did the capstone's 99 percent loyalty model actually measure?

Data: Kaggle's Airline Passenger Satisfaction survey (train.csv 103,904 rows,
test.csv 25,976 rows). Target, as in the capstone: Customer Type == "Loyal Customer".
Kaggle's own test file is the holdout throughout.
"""
import json
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

SEED = 42
SERVICE = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location",
           "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service",
           "Leg room service", "Baggage handling", "Checkin service", "Inflight service", "Cleanliness"]

def prep(path):
    d = pd.read_csv(path, index_col=0)
    d["loyal"] = (d["Customer Type"] == "Loyal Customer").astype(int)
    d["business_travel"] = (d["Type of Travel"] == "Business travel").astype(int)
    d["class_code"] = d["Class"].map({"Eco": 0, "Eco Plus": 1, "Business": 2})
    d["male"] = (d.Gender == "Male").astype(int)
    d["Arrival Delay in Minutes"] = d["Arrival Delay in Minutes"].fillna(d["Departure Delay in Minutes"])
    return d
tr, te = prep("data/raw/train.csv"), prep("data/raw/test.csv")
CAPSTONE = SERVICE + ["Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes", "Age", "class_code", "business_travel", "male"]
gz = lambda d: d[d.Age <= 27]

out = {"n_train": len(tr), "n_test": len(te), "loyal_rate": tr.loyal.mean(), "genz_loyal_rate": gz(tr).loyal.mean()}

# 1. Reproduce: Gen Z rows, capstone features, random 80/20 split, XGBoost as configured in the notebook.
g = gz(tr)
a, b = train_test_split(g, test_size=0.2, stratify=g.loyal, random_state=SEED)
xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=SEED, eval_metric="logloss")
xgb.fit(a[CAPSTONE], a.loyal); p = xgb.predict_proba(b[CAPSTONE])[:, 1]
out["reproduce"] = {"auc": roc_auc_score(b.loyal, p), "accuracy": accuracy_score(b.loyal, p > 0.5), "n_test": len(b)}
print("reproduced capstone:", {k: round(v, 4) for k, v in out["reproduce"].items()})

# 2. The shortcut: how is Customer Type spread across travel type and age?
ct = pd.crosstab(tr["Type of Travel"], tr["Customer Type"], normalize="index")
out["crosstab_travel"] = ct.round(4).to_dict(orient="index")
print(ct.round(3))
age_band = pd.cut(tr.Age, [0, 27, 43, 59, 100], labels=["Gen Z (27 and under)", "Millennial", "Gen X", "Boomer"])
out["loyal_by_generation"] = tr.groupby(age_band, observed=True).loyal.mean().round(4).to_dict()
print(out["loyal_by_generation"])

def fit_score(cols, model="lgb", train=tr, test=te):
    if model == "logit":
        m = LogisticRegression(max_iter=2000).fit(train[cols], train.loyal); p = m.predict_proba(test[cols])[:, 1]
    else:
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
        m.fit(train[cols], train.loyal); p = m.predict_proba(test[cols])[:, 1]
    return {"auc": roc_auc_score(test.loyal, p), "accuracy": accuracy_score(test.loyal, p > 0.5)}, m

# 3. Ablation on Kaggle's holdout (all ages and Gen Z only)
sets = {
    "Majority guess (everyone loyal)": None,
    "Age only": ["Age"],
    "Age + travel type": ["Age", "business_travel"],
    "Age + travel type + class": ["Age", "business_travel", "class_code"],
    "All capstone inputs": CAPSTONE,
    "Service ratings only (14)": SERVICE,
    "Everything except age, travel type, class": [c for c in CAPSTONE if c not in ("Age", "business_travel", "class_code")],
}
ablation = []
for name, cols in sets.items():
    row = {"inputs": name}
    for scope, (trn, tst) in {"all": (tr, te), "genz": (gz(tr), gz(te))}.items():
        if cols is None:
            row[scope] = {"auc": 0.5, "accuracy": float(max(tst.loyal.mean(), 1 - tst.loyal.mean()))}
        else:
            row[scope], _ = fit_score(cols, train=trn, test=tst)
    ablation.append(row)
    print(f"{name:44s} all AUC {row['all']['auc']:.3f} acc {row['all']['accuracy']:.3f} | Gen Z AUC {row['genz']['auc']:.3f} acc {row['genz']['accuracy']:.3f}")
out["ablation"] = ablation

# 4. With the shortcut held fixed: among personal travelers only, do service ratings separate loyal from disloyal?
pers_tr, pers_te = tr[tr.business_travel == 0], te[te.business_travel == 0]
biz_tr, biz_te = tr[tr.business_travel == 1], te[te.business_travel == 1]
out["within_travel"] = {}
for nm, (a2, b2) in {"business": (biz_tr, biz_te), "personal": (pers_tr, pers_te)}.items():
    s, _ = fit_score(SERVICE + ["Flight Distance", "class_code"], train=a2, test=b2)
    out["within_travel"][nm] = {**s, "loyal_rate": float(b2.loyal.mean()), "n_test": len(b2)}
    print(f"service-only model within {nm} travelers: AUC {s['auc']:.3f}, loyal rate {b2.loyal.mean():.3f}")

# 5. What the service-only model leans on (LightGBM gain importance, normalized)
_, m = fit_score(SERVICE)
imp = pd.Series(m.booster_.feature_importance("gain"), index=SERVICE); imp = (imp / imp.sum()).sort_values(ascending=False)
out["service_importance"] = imp.round(4).to_dict()
print("service-only drivers:", imp.head(6).round(3).to_dict())
gaps = (tr[tr.loyal == 1][SERVICE].mean() - tr[tr.loyal == 0][SERVICE].mean()).sort_values(ascending=False)
out["rating_gap_loyal_minus_disloyal"] = gaps.round(3).to_dict()
print("mean rating gap loyal - disloyal:", gaps.round(2).head(6).to_dict())
json.dump(out, open("results/loyalty.json", "w"), indent=2, default=float)

# 6. Every combination of five input groups, scored on the holdout, for the page's interactive build-up.
from itertools import combinations
GROUPS = {"age": ["Age"], "travel": ["business_travel"], "cls": ["class_code"], "service": SERVICE,
          "other": ["Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes", "male"]}
combos = {}
keys = list(GROUPS)
for r in range(1, len(keys) + 1):
    for combo in combinations(keys, r):
        cols = sum((GROUPS[k] for k in combo), [])
        s_all, _ = fit_score(cols)
        s_gz, _ = fit_score(cols, train=gz(tr), test=gz(te))
        combos["+".join(combo)] = {"all": s_all, "genz": s_gz}
print(f"{len(combos)} combinations scored")
out["combos"] = combos
out["na_departure_convenience_business"] = {
    "disloyal_share_when_0": float(1 - tr[(tr.business_travel == 1) & (tr["Departure/Arrival time convenient"] == 0)].loyal.mean()),
    "disloyal_share_overall": float(1 - tr[tr.business_travel == 1].loyal.mean())}
json.dump(out, open("results/loyalty.json", "w"), indent=2, default=float)

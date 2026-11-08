"""Compact data for the American Airlines page's audit section. Usage: export_page_data.py OUT.json"""
import json, sys
d = json.load(open("results/loyalty.json"))
abl = {r["inputs"]: r for r in d["ablation"]}
out = {
    "reproduce": d["reproduce"],
    "combos": {k: [round(v["all"]["auc"], 3), round(v["genz"]["auc"], 3)] for k, v in d["combos"].items()},
    "majorityAccuracy": [abl["Majority guess (everyone loyal)"]["all"]["accuracy"], abl["Majority guess (everyone loyal)"]["genz"]["accuracy"]],
    "loyalPersonal": d["crosstab_travel"]["Personal Travel"]["Loyal Customer"],
    "loyalBusiness": d["crosstab_travel"]["Business travel"]["Loyal Customer"],
    "byGeneration": d["loyal_by_generation"],
    "withinBusinessAuc": d["within_travel"]["business"]["auc"],
    "naDisloyal": d["na_departure_convenience_business"],
    "nTest": d["n_test"],
}
json.dump(out, open(sys.argv[1], "w"), separators=(",", ":"), default=float)
print("ok", out["reproduce"], round(out["loyalPersonal"], 4), round(out["withinBusinessAuc"], 4))

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ART = Path("artifacts")
MODEL = ART / "model_calibrated.pkl"
THR   = ART / "thresholds.json"
HOLD  = Path("data/processed/test_holdout.csv")
TARGET, IDCOL = "Default", "LoanID"

def decide(p, t_approve, t_reject):
    if p < t_approve: return "Approved"
    if p > t_reject:  return "Rejected"
    return "Review"

def predict_by_id(loan_id: str):
    model = joblib.load(MODEL)
    thr = json.loads(Path(THR).read_text())
    df = pd.read_csv(HOLD)
    row = df[df[IDCOL] == loan_id]
    if row.empty: 
        return {"error": f"{loan_id} no está en test_holdout"}
    X = row.drop(columns=[TARGET, IDCOL])
    proba = float(model.predict_proba(X)[:,1][0])
    return {
        "LoanID": loan_id,
        "proba_default": proba,
        "decision": decide(proba, thr["t_approve"], thr["t_reject"]),
        "thresholds": thr
    }

if __name__ == "__main__":
    import sys
    print(predict_by_id(sys.argv[1] if len(sys.argv)>1 else ""))
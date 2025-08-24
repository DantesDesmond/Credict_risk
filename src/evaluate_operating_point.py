# src/evaluate_operating_point.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import argparse
from sklearn.metrics import roc_auc_score

ART = Path("artifacts")
MODEL = ART / "model_calibrated.pkl"
THR_DEFAULT = ART / "thresholds.json"
THR_POLICY = ART / "thresholds_policy.json"
HOLD = Path("data/processed/test_holdout.csv")
OUT = ART / "operating_report"; OUT.mkdir(parents=True, exist_ok=True)

TARGET, IDCOL = "Default", "LoanID"

def _load_thresholds():
    if THR_POLICY.exists():
        thr = json.loads(THR_POLICY.read_text(encoding="utf-8")); src = THR_POLICY.name
    else:
        thr = json.loads(THR_DEFAULT.read_text(encoding="utf-8")); src = THR_DEFAULT.name
    return thr, src

def decide_codes(proba, t_low, t_high):
    # 0=Approved,1=Review,2=Rejected
    return np.where(proba < t_low, 0, np.where(proba > t_high, 2, 1))

def apply_guardrails(df: pd.DataFrame, dec: np.ndarray, params: dict) -> np.ndarray:
    """
    Mueve a REVIEW (1) ciertos 'Approved' (0) según reglas de riesgo.
    Reglas por defecto (ajustables por CLI):
      R1: DTIRatio >= dti_hi AND CreditScore <= cs_low
      R2: MonthsEmployed < min_months
      R3: Income < income_lo AND LoanAmount > loan_hi
      R4: (HasCoSigner == No) AND (LoanPurpose in {Business, Other}) AND InterestRate > rate_hi
    """
    d = dec.copy()
    # Helpers robustos
    def col_num(name, default=0):
        c = df.get(name)
        if c is None: return pd.Series(default, index=df.index, dtype=float)
        return pd.to_numeric(c, errors="coerce").fillna(default)

    def col_str(name):
        c = df.get(name)
        if c is None: return pd.Series("", index=df.index, dtype=object)
        s = c.astype(str).str.strip().str.lower()
        # normalizar yes/no/1/0
        s = s.replace({"true":"yes","false":"no","1":"yes","0":"no"})
        return s

    # Columnas usadas
    dti   = col_num("DTIRatio")
    cs    = col_num("CreditScore")
    me    = col_num("MonthsEmployed")
    inc   = col_num("Income")
    loan  = col_num("LoanAmount")
    ir    = col_num("InterestRate")
    cos   = col_str("HasCoSigner")
    purp  = col_str("LoanPurpose")

    # Parámetros
    dti_hi    = params.get("dti_hi", 0.45)
    cs_low    = params.get("cs_low", 640)
    min_mo    = params.get("min_months_employed", 6)
    income_lo = params.get("income_lo", 15000)
    loan_hi   = params.get("loan_hi", 80000)
    rate_hi   = params.get("rate_hi", 21.0)
    risky_purposes = set([p.strip().lower() for p in params.get("risky_purposes", ["business","other"])])

    # Reglas (solo afectan Approved -> Review)
    approved_mask = (d == 0)
    r1 = (dti >= dti_hi) & (cs <= cs_low)
    r2 = (me < min_mo)
    r3 = (inc < income_lo) & (loan > loan_hi)
    r4 = (cos.isin(["no","false"])) & (purp.isin(risky_purposes)) & (ir > rate_hi)

    override_to_review = approved_mask & (r1 | r2 | r3 | r4)
    d[override_to_review] = 1
    return d

def summarize(df: pd.DataFrame, y: np.ndarray, proba: np.ndarray, dec: np.ndarray):
    n = len(df)
    out = {}
    for lab, code in [("Approved",0),("Review",1),("Rejected",2)]:
        mask = (dec == code)
        cnt = int(mask.sum())
        br  = float((y[mask] == 1).mean()) if cnt>0 else float("nan")
        out[lab] = {"count": cnt, "rate": round(cnt/n, 4), "bad_rate": None if np.isnan(br) else round(br, 4)}
    base_br = float((y == 1).mean())
    auc = float(roc_auc_score(y, proba))
    return {"auc_holdout": auc, "base_bad_rate": base_br, "by_decision": out}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_guardrails", action="store_true")
    ap.add_argument("--dti_hi", type=float, default=0.45)
    ap.add_argument("--cs_low", type=float, default=640)
    ap.add_argument("--min_months_employed", type=int, default=6)
    ap.add_argument("--income_lo", type=float, default=15000)
    ap.add_argument("--loan_hi", type=float, default=80000)
    ap.add_argument("--rate_hi", type=float, default=21.0)
    ap.add_argument("--risky_purposes", type=str, default="business,other")
    args = ap.parse_args()

    model = joblib.load(MODEL)
    thr, thr_src = _load_thresholds()
    df = pd.read_csv(HOLD).copy()

    X = df.drop(columns=[TARGET, IDCOL])
    y = df[TARGET].values
    proba = model.predict_proba(X)[:,1]

    dec = decide_codes(proba, thr["t_approve"], thr["t_reject"])

    meta = {"thresholds_source": thr_src, "thresholds": thr, "use_guardrails": bool(args.use_guardrails)}

    if args.use_guardrails:
        params = {
            "dti_hi": args.dti_hi,
            "cs_low": args.cs_low,
            "min_months_employed": args.min_months_employed,
            "income_lo": args.income_lo,
            "loan_hi": args.loan_hi,
            "rate_hi": args.rate_hi,
            "risky_purposes": [s.strip() for s in args.risky_purposes.split(",") if s.strip()],
        }
        dec = apply_guardrails(df, dec, params)
        meta["guardrails"] = params

    # Resumen y artefactos
    summary = summarize(df, y, proba, dec)
    (OUT / "summary.json").write_text(json.dumps({**summary, **meta}, indent=2), encoding="utf-8")

    # Detalle por deciles (útil para QA)
    df_out = df[[IDCOL, TARGET]].copy()
    df_out["proba"] = proba
    df_out["decision"] = np.where(dec==0,"Approved", np.where(dec==1,"Review","Rejected"))
    df_out.to_csv(OUT / "decisions_holdout.csv", index=False)

    # Histograma de scores
    hist, edges = np.histogram(proba, bins=50, range=(0,1))
    np.savez(OUT / "score_hist.npz", hist=hist, edges=edges,
             t_approve=thr["t_approve"], t_reject=thr["t_reject"])

    print(json.dumps({**summary, **meta, "artifacts": {
        "summary": str((OUT/"summary.json").resolve()),
        "decisions_csv": str((OUT/"decisions_holdout.csv").resolve())
    }}, indent=2))

if __name__ == "__main__":
    main()
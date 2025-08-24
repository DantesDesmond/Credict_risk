# src/policy_search.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import argparse

ART = Path("artifacts")
MODEL = ART / "model_calibrated.pkl"
HOLD  = Path("data/processed/test_holdout.csv")
OUT   = ART / "thresholds_policy.json"

TARGET, IDCOL = "Default", "LoanID"

def evaluate(y, proba, t_low, t_high):
    """
    Decisiones: 0=Approved, 1=Review, 2=Rejected
    """
    dec = np.where(proba < t_low, 0, np.where(proba > t_high, 2, 1))
    n = len(y)
    p_apr = float((dec == 0).mean())
    p_rev = float((dec == 1).mean())
    p_rej = float((dec == 2).mean())

    # métricas de pureza/captura
    br_apr = float((y[dec == 0] == 1).mean()) if (dec == 0).any() else np.nan
    total_bad = int((y == 1).sum())
    total_good = int((y == 0).sum())
    bad_approved = int(((y == 1) & (dec == 0)).sum())
    good_approved = int(((y == 0) & (dec == 0)).sum())

    recall_bad_not_approved = float(1.0 - bad_approved / total_bad) if total_bad > 0 else np.nan
    captured_bad_in_reject  = float(((y == 1) & (dec == 2)).sum() / total_bad) if total_bad > 0 else np.nan
    good_approval_rate      = float(good_approved / total_good) if total_good > 0 else np.nan

    return {
        "t_approve": float(t_low), "t_reject": float(t_high),
        "p_apr": p_apr, "p_rev": p_rev, "p_rej": p_rej,
        "bad_rate_approved": None if np.isnan(br_apr) else float(br_apr),
        "bad_approved": bad_approved, "bad_total": total_bad,
        "recall_bad_not_approved": recall_bad_not_approved,
        "captured_bad_in_reject": captured_bad_in_reject,
        "good_approval_rate": good_approval_rate
    }

def score_candidate(res, targets, tol, max_bad_apr):
    """
    Puntuación: prioriza no aprobar malos; luego cumplir proporciones y maximizar
    buenos aprobados / malos rechazados.
    """
    # penalizaciones
    dist = (abs(res["p_apr"] - targets["approve"]) +
            abs(res["p_rev"] - targets["review"])  +
            abs(res["p_rej"] - targets["reject"]))

    br = res["bad_rate_approved"]
    br_violation = 0.0 if (br is not None and br <= max_bad_apr) else (1.0 if br is None else (br - max_bad_apr))

    # objetivo
    score = (
        - res["bad_approved"] * 1_000_000                # 1) ¡no aprobar malos!
        - br_violation * 100_000                         # 2) respetar el tope de bad rate en aprobados
        - max(0.0, dist - 3*tol) * 10_000               # 3) acercarse a 85/5/10 (tolerancia)
        + (res["captured_bad_in_reject"] or 0.0) * 5_000 # 4) concentrar malos en Rechazo
        + (res["good_approval_rate"]   or 0.0) * 1_000   # 5) aprobar buenos
    )
    return float(score), float(dist)

def search_thresholds_targets(
    y, proba,
    approve_target=0.85, review_target=0.05, reject_target=0.10,
    tolerance=0.015,                # ±1.5 pp
    max_bad_rate_approved=0.08,     # tope de bad rate en aprobados (ajustable)
    grid_low=(0.01, 0.50, 0.001),   # barrido t_approve
    grid_high=(0.05, 0.95, 0.001)   # barrido t_reject
):
    targets = {"approve": approve_target, "review": review_target, "reject": reject_target}
    low_start, low_end, low_step = grid_low
    high_start, high_end, high_step = grid_high

    best_strict = None
    best_relaxed = None

    # 1) Solución estricta: cumplir proporciones dentro de tolerancia y tope de bad rate
    for t_low in np.arange(low_start, low_end, low_step):
        for t_high in np.arange(high_start, high_end, high_step):
            if t_low >= t_high:
                continue
            res = evaluate(y, proba, t_low, t_high)
            br_ok = (res["bad_rate_approved"] is not None) and (res["bad_rate_approved"] <= max_bad_rate_approved)
            within = (
                abs(res["p_apr"] - targets["approve"]) <= tolerance and
                abs(res["p_rev"] - targets["review"])  <= tolerance and
                abs(res["p_rej"] - targets["reject"])  <= tolerance
            )
            if br_ok and within:
                s, dist = score_candidate(res, targets, tolerance, max_bad_rate_approved)
                cand = {**res, "score": s, "distance": dist, "note": "strict_ok",
                        "targets": targets, "tolerance": tolerance, "max_bad_rate_approved": max_bad_rate_approved}
                if (best_strict is None) or (cand["score"] > best_strict["score"]):
                    best_strict = cand

    if best_strict is not None:
        return best_strict

    # 2) Relajado: exige tope de bad rate pero permite desviaciones mínimas de proporciones, minimizando la distancia total
    for t_low in np.arange(low_start, low_end, low_step):
        for t_high in np.arange(high_start, high_end, high_step):
            if t_low >= t_high:
                continue
            res = evaluate(y, proba, t_low, t_high)
            br_ok = (res["bad_rate_approved"] is not None) and (res["bad_rate_approved"] <= max_bad_rate_approved)
            if not br_ok:
                continue
            s, dist = score_candidate(res, targets, tolerance, max_bad_rate_approved)
            cand = {**res, "score": s, "distance": dist, "note": "relaxed_targets",
                    "targets": targets, "tolerance": tolerance, "max_bad_rate_approved": max_bad_rate_approved}
            if (best_relaxed is None) or (cand["score"] > best_relaxed["score"]):
                best_relaxed = cand

    if best_relaxed is not None:
        return best_relaxed

    # 3) Fallback muy conservador por cuantiles (garantiza pureza pero no proporciones)
    t_low = float(np.quantile(proba, targets["approve"]))
    t_high = float(np.quantile(proba, targets["approve"] + targets["review"]))
    res = evaluate(y, proba, t_low, t_high)
    s, dist = score_candidate(res, targets, tolerance, max_bad_rate_approved)
    return {**res, "score": s, "distance": dist, "note": "fallback_quantiles",
            "targets": targets, "tolerance": tolerance, "max_bad_rate_approved": max_bad_rate_approved}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approve_target", type=float, default=0.85)
    ap.add_argument("--review_target",  type=float, default=0.05)
    ap.add_argument("--reject_target",  type=float, default=0.10)
    ap.add_argument("--tolerance",      type=float, default=0.015)  # ±1.5 pp
    ap.add_argument("--max_bad_rate_approved", type=float, default=0.08)
    ap.add_argument("--grid_low_start",  type=float, default=0.01)
    ap.add_argument("--grid_low_end",    type=float, default=0.50)
    ap.add_argument("--grid_low_step",   type=float, default=0.001)
    ap.add_argument("--grid_high_start", type=float, default=0.05)
    ap.add_argument("--grid_high_end",   type=float, default=0.95)
    ap.add_argument("--grid_high_step",  type=float, default=0.001)
    args = ap.parse_args()

    model = joblib.load(MODEL)
    df = pd.read_csv(HOLD)
    Xh = df.drop(columns=[TARGET, IDCOL])
    y = df[TARGET].values
    proba = model.predict_proba(Xh)[:, 1]

    best = search_thresholds_targets(
        y, proba,
        approve_target=args.approve_target,
        review_target=args.review_target,
        reject_target=args.reject_target,
        tolerance=args.tolerance,
        max_bad_rate_approved=args.max_bad_rate_approved,
        grid_low=(args.grid_low_start, args.grid_low_end, args.grid_low_step),
        grid_high=(args.grid_high_start, args.grid_high_end, args.grid_high_step),
    )

    # Guarda JSON con metadata de la política y los resultados
    OUT.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))

if __name__ == "__main__":
    main()
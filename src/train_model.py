# src/train_model.py
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

RANDOM_STATE = 42

DATA_TRAIN = Path("data/processed/train.csv")
DATA_TEST  = Path("data/processed/test_holdout.csv")
OUT_DIR    = Path("artifacts"); OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Default"
IDCOL  = "LoanID"

# columnas según tu dataset
NUM_COLS = [
    "Age","Income","LoanAmount","CreditScore","MonthsEmployed",
    "NumCreditLines","InterestRate","LoanTerm","DTIRatio"
]
CAT_COLS = ["Education","EmploymentType","MaritalStatus","HasMortgage","HasDependents","LoanPurpose","HasCoSigner"]

def ks_score(y_true, y_proba):
    """Kolmogorov–Smirnov para binario (mayor es mejor)."""
    # percentiles para buenos y malos
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return np.max(np.abs(tpr - fpr))

def fit_and_calibrate(X, y, base_estimator):
    """
    Ajusta pipeline + calibración (isotonic si datos suficientes; si no, sigmoid).
    Devuelve (pipeline_calibrado, cv_metrics dict, oof_probas).
    """
    numeric = Pipeline(steps=[("scaler", StandardScaler())])
    categorical = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore"))])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, [c for c in NUM_COLS if c in X.columns]),
            ("cat", categorical, [c for c in CAT_COLS if c in X.columns]),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    pipe = Pipeline(steps=[("pre", pre), ("clf", base_estimator)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # probas out-of-fold para métricas honestas
    oof_proba = cross_val_predict(
        pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y, oof_proba)),
        "pr_auc": float(average_precision_score(y, oof_proba)),
        "brier": float(brier_score_loss(y, oof_proba)),
        "ks": float(ks_score(y, oof_proba)),
    }

    method = "isotonic" if len(y) > 20000 else "sigmoid"
    calib = CalibratedClassifierCV(pipe, method=method, cv=5)
    calib.fit(X, y)

    return calib, metrics, oof_proba

def find_operational_thresholds(y, proba, approve_target=0.80, review_target=0.10, tolerance=0.03, max_bad_rate_approved=0.10):
    """
    Busca par (t_approve, t_reject) que cumpla %aprob≈80, %rev≈10, %rech≈10
    y bad rate en aprobados ≤ 10%. Si no encuentra, devuelve mejor factible.
    """
    y = np.asarray(y)
    proba = np.asarray(proba)
    best = None
    for t_a in np.linspace(0.02, 0.30, 57):          # umbral bajo para Aprobado
        for t_r in np.linspace(0.40, 0.95, 111):     # umbral alto para Rechazo
            if t_a >= t_r: 
                continue
            dec = np.where(proba < t_a, 0, np.where(proba > t_r, 2, 1))
            # 0=Approved,1=Review,2=Reject
            p_apr = (dec == 0).mean()
            p_rev = (dec == 1).mean()
            p_rej = (dec == 2).mean()
            if (abs(p_apr - approve_target) <= tolerance and
                abs(p_rev - review_target)  <= tolerance and
                abs(p_rej - (1-approve_target-review_target)) <= tolerance):
                # bad rate en aprobados
                br_approved = (y[(dec==0)]==1).mean() if (dec==0).any() else 1.0
                if br_approved <= max_bad_rate_approved:
                    # score: maximiza distancia a bad rate permitido y ROC
                    score = (approve_target - abs(p_apr-approve_target)) + (review_target - abs(p_rev-review_target)) + (1-abs(br_approved-max_bad_rate_approved))
                    cand = {"t_approve":float(t_a),"t_reject":float(t_r),
                            "p_apr":float(p_apr),"p_rev":float(p_rev),"p_rej":float(p_rej),
                            "bad_rate_approved":float(br_approved),"score":float(score)}
                    if (best is None) or (cand["score"] > best["score"]):
                        best = cand
    # fallback: relajar si no encontró nada
    if best is None:
        t = np.quantile(proba, 0.80)
        u = np.quantile(proba, 0.90)
        dec = np.where(proba < t, 0, np.where(proba > u, 2, 1))
        br_approved = (y[(dec==0)]==1).mean() if (dec==0).any() else 1.0
        best = {"t_approve":float(t),"t_reject":float(u),
                "p_apr":float((dec==0).mean()),"p_rev":float((dec==1).mean()),
                "p_rej":float((dec==2).mean()),"bad_rate_approved":float(br_approved),
                "score":0.0,"note":"fallback_quantiles"}
    return best

def evaluate_on_test(model, df_test):
    Xt = df_test.drop(columns=[TARGET, IDCOL])
    yt = df_test[TARGET].values
    proba = model.predict_proba(Xt)[:,1]
    return {
        "roc_auc": float(roc_auc_score(yt, proba)),
        "pr_auc": float(average_precision_score(yt, proba)),
        "brier": float(brier_score_loss(yt, proba)),
        "ks": float(ks_score(yt, proba)),
    }, yt, proba

def main():
    train = pd.read_csv(DATA_TRAIN)
    test  = pd.read_csv(DATA_TEST)

    X = train.drop(columns=[TARGET, IDCOL])
    y = train[TARGET].values

    # Modelos base
    models = {
        "logreg": LogisticRegression(max_iter=1000, class_weight=None, random_state=RANDOM_STATE),
        "hgb": HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    }

    results = {}
    fitted = {}

    for name, est in models.items():
        clf, cv_metrics, oof_proba = fit_and_calibrate(X, y, est)
        results[name] = {"cv": cv_metrics}
        fitted[name] = {"model": clf, "oof_proba": oof_proba}

    # Elegir por ROC AUC (puedes cambiar el criterio)
    best_name = max(results, key=lambda n: results[n]["cv"]["roc_auc"])
    best_model = fitted[best_name]["model"]
    oof_proba  = fitted[best_name]["oof_proba"]

    # Umbrales operativos en train (oof)
    thr = find_operational_thresholds(
        y, oof_proba, approve_target=0.80, review_target=0.10, tolerance=0.03, max_bad_rate_approved=0.10
    )

    # Eval test
    test_metrics, y_test, proba_test = evaluate_on_test(best_model, test)

    # ROC / PR para Streamlit
    fpr, tpr, roc_th = roc_curve(y_test, proba_test)
    prec, rec, pr_th = precision_recall_curve(y_test, proba_test)

    # Guardar artefactos
    joblib.dump(best_model, OUT_DIR / "model_calibrated.pkl")
    with open(OUT_DIR / "thresholds.json","w") as f:
        json.dump(thr, f, indent=2)
    with open(OUT_DIR / "metrics.json","w") as f:
        json.dump({
            "best_model": best_name,
            "cv": results[best_name]["cv"],
            "test": test_metrics
        }, f, indent=2)
    np.savez(OUT_DIR / "curves_test.npz", fpr=fpr, tpr=tpr, roc_th=roc_th, prec=prec, rec=rec, pr_th=pr_th)

    print("== Entrenamiento finalizado ==")
    print(f"✔ Mejor modelo: {best_name}")
    print("CV metrics:", results[best_name]["cv"])
    print("Test metrics:", test_metrics)
    print("Umbrales:", thr)
    print(f"Artefactos: {OUT_DIR.resolve()}/(model_calibrated.pkl, thresholds.json, metrics.json, curves_test.npz)")

if __name__ == "__main__":
    main()
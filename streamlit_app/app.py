# streamlit_app/app.py
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------- Rutas robustas (soporta ejecutar desde streamlit_app/ o desde raíz) ----------
def find_project_root():
    here = Path(__file__).resolve().parent
    # Caso 1: estás en streamlit_app/
    cand = here.parent
    # Verifica que existan las carpetas esperadas en el root
    if (cand / "artifacts").exists() and (cand / "data" / "processed").exists():
        return cand
    # Caso 2: ejecutas desde raíz y __file__ ya es raíz/streamlit_app/app.py
    return Path.cwd()

ROOT = find_project_root()
ART = ROOT / "artifacts"
DATA = ROOT / "data" / "processed"

MODEL = ART / "model_calibrated.pkl"
THR_DEFAULT = ART / "thresholds.json"          # 80/10/10
THR_POLICY  = ART / "thresholds_policy.json"   # política conservadora (si existe)
CURV  = ART / "curves_test.npz"
METR  = ART / "metrics.json"
HOLD  = DATA / "test_holdout.csv"

TARGET, IDCOL = "Default", "LoanID"

# ===================== UI CONFIG =====================
st.set_page_config(page_title="Credit Risk – Scoring & Decision", layout="wide")
st.title("Credit Risk – Scoring & Decision")


# ---------- Utilidades ----------
def decide(p, t_low, t_high):
    if p < t_low:
        return "Approved"
    if p > t_high:
        return "Rejected"
    return "Review"


@st.cache_resource
def load_artifacts():
    # Modelo
    if not MODEL.exists():
        raise FileNotFoundError(f"No existe el modelo: {MODEL}")
    model = joblib.load(MODEL)

    # Umbrales (policy > default)
    if THR_POLICY.exists():
        thr = json.loads(THR_POLICY.read_text(encoding="utf-8"))
        thr_source = THR_POLICY.name
    elif THR_DEFAULT.exists():
        thr = json.loads(THR_DEFAULT.read_text(encoding="utf-8"))
        thr_source = THR_DEFAULT.name
    else:
        raise FileNotFoundError("No se encontró thresholds_policy.json ni thresholds.json en artifacts/")

    # Curvas y métricas
    if not CURV.exists():
        raise FileNotFoundError(f"No existe archivo de curvas: {CURV}")
    curves = np.load(CURV)

    if not METR.exists():
        raise FileNotFoundError(f"No existe archivo de métricas: {METR}")
    metrics = json.loads(METR.read_text(encoding="utf-8"))

    # Holdout
    if not HOLD.exists():
        raise FileNotFoundError(f"No existe holdout: {HOLD}")
    df_holdout = pd.read_csv(HOLD)

    return model, thr, thr_source, curves, metrics, df_holdout


def by_decision_table_from_proba(df_holdout: pd.DataFrame, proba_all: np.ndarray, thr: dict):
    decisions = np.where(
        proba_all < thr["t_approve"], "Approved",
        np.where(proba_all > thr["t_reject"], "Rejected", "Review")
    )
    tmp = df_holdout.copy()
    tmp["proba"] = proba_all
    tmp["decision"] = decisions
    by_dec = (
        tmp.groupby("decision")
           .agg(
               rows=("decision", "size"),
               rate=("decision", lambda s: len(s) / len(tmp)),
               bad_rate=(TARGET, "mean"),
           )
           .reindex(["Approved", "Review", "Rejected"])
    )
    return tmp, by_dec


# ============ Carga de artefactos ============
try:
    model, thr, thr_source, curves, metrics, df_holdout = load_artifacts()
except Exception as e:
    st.error(f"No pude cargar los artefactos. Verifica rutas y ejecuta el entrenamiento/política primero.\n{e}")
    st.stop()

# Validar claves mínimas en thresholds
for k in ("t_approve", "t_reject"):
    if k not in thr:
        st.error(f"El archivo de umbrales '{thr_source}' no contiene la clave requerida: {k}")
        st.stop()

# ---- Panel superior de estado ----
note_text = (
    "Usando **thresholds_policy.json** (política conservadora: minimizar malos aprobados)."
    if thr_source == THR_POLICY.name
    else "Usando **thresholds.json** (política 80/10/10 con bad rate en aprobados ≤ 10%)."
)
st.info(f"**Umbrales activos:** {note_text}")

# ---- Panel de métricas globales ----
colA, colB, colC = st.columns(3)
with colA:
    st.metric("Mejor modelo (CV)", metrics.get("best_model", "—"))
with colB:
    st.metric("ROC AUC (CV)", f"{metrics.get('cv', {}).get('roc_auc', 0):.3f}")
with colC:
    st.metric("ROC AUC (Test)", f"{metrics.get('test', {}).get('roc_auc', 0):.3f}")

st.caption("El modelo está calibrado. Las decisiones dependen de los umbrales activos mostrados arriba.")

# ---- Visual: ROC del holdout ----
st.subheader("ROC (holdout)")
fpr, tpr = curves["fpr"], curves["tpr"]
fig_roc = plt.figure()
plt.plot(fpr, tpr, label="ROC")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve (holdout)")
st.pyplot(fig_roc)

# ---- Distribución de probabilidades + umbrales ----
st.subheader("Distribución de probabilidades (holdout)")
Xh = df_holdout.drop(columns=[TARGET, IDCOL])
proba_all = model.predict_proba(Xh)[:, 1]

fig_hist = plt.figure()
hist = np.histogram(proba_all, bins=50, range=(0, 1))
plt.bar(hist[1][:-1], hist[0], width=np.diff(hist[1]), align="edge")
plt.axvline(thr["t_approve"], linestyle="--")
plt.axvline(thr["t_reject"], linestyle="--")
plt.xlabel("Probabilidad de default")
plt.ylabel("Frecuencia")
st.pyplot(fig_hist)

# ---- Resumen por decisión (holdout) ----
tmp, by_dec = by_decision_table_from_proba(df_holdout, proba_all, thr)
st.subheader("Operación (holdout) por bucket de decisión")
st.dataframe(by_dec.style.format({"rate": "{:.2%}", "bad_rate": "{:.2%}"}))

# ---- Búsqueda por LoanID ----
st.markdown("---")
st.subheader("Consulta individual por LoanID (holdout)")
loan_id = st.text_input("Ingresa un LoanID que esté en el holdout:", value="")
btn = st.button("Evaluar")

if btn and loan_id.strip():
    # Convertir a str para evitar desajustes de tipo
    lid = str(loan_id).strip()
    row = df_holdout[df_holdout[IDCOL].astype(str) == lid]
    if row.empty:
        st.error("LoanID no encontrado en el holdout.")
    else:
        Xi = row.drop(columns=[TARGET, IDCOL])
        p = float(model.predict_proba(Xi)[:, 1][0])
        decision = decide(p, thr["t_approve"], thr["t_reject"])

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Probabilidad de default", f"{p:.3f}")
        with c2:
            st.metric("Decisión", decision)

        st.write("**Umbrales operativos actuales (archivo cargado):**")
        st.json({"source": thr_source, **thr})

        # Mostrar features principales de la fila (sin target ni id)
        st.write("**Características del caso:**")
        # Mostrar en formato columna: variable | valor
        pair_df = Xi.T.copy()
        pair_df.columns = ["value"]
        st.dataframe(pair_df)

# Footer
st.markdown("---")
st.caption(f"Rutas detectadas → ROOT: `{ROOT}` | ART: `{ART}` | DATA: `{DATA}`")

with st.expander("Guardrails (reglas) sobre Aprobados", expanded=False):
    use_gr = st.checkbox("Aplicar guardrails a los Aprobados", value=False)
    dti_hi = st.number_input("DTI alto ≥", value=0.45, step=0.01)
    cs_low = st.number_input("CreditScore bajo ≤", value=640, step=10)
    min_mo = st.number_input("Meses de empleo <", value=6, step=1)
    income_lo = st.number_input("Ingreso mensual <", value=15000, step=500)
    loan_hi = st.number_input("LoanAmount >", value=80000, step=5000)
    rate_hi = st.number_input("InterestRate >", value=21.0, step=0.5)
    risky_purposes = st.text_input("Propósitos riesgosos (coma)", value="business,other").lower().split(",")

def apply_guardrails_streamlit(df_holdout, dec, params):
    tmp = df_holdout.copy()
    def col_num(name, default=0):
        c = tmp.get(name)
        if c is None: return pd.Series(default, index=tmp.index, dtype=float)
        return pd.to_numeric(c, errors="coerce").fillna(default)
    def col_str(name):
        c = tmp.get(name)
        if c is None: return pd.Series("", index=tmp.index, dtype=object)
        s = c.astype(str).str.strip().str.lower()
        s = s.replace({"true":"yes","false":"no","1":"yes","0":"no"})
        return s

    dti, cs = col_num("DTIRatio"), col_num("CreditScore")
    me, inc, loan = col_num("MonthsEmployed"), col_num("Income"), col_num("LoanAmount")
    ir, cos = col_num("InterestRate"), col_str("HasCoSigner")
    purp = col_str("LoanPurpose")

    approved_mask = (dec == 0)
    r1 = (dti >= params["dti_hi"]) & (cs <= params["cs_low"])
    r2 = (me < params["min_months_employed"])
    r3 = (inc < params["income_lo"]) & (loan > params["loan_hi"])
    r4 = (cos.isin(["no","false"])) & (purp.isin([p.strip() for p in params["risky_purposes"]])) & (ir > params["rate_hi"])
    override = approved_mask & (r1 | r2 | r3 | r4)

    dec2 = dec.copy()
    dec2[override] = 1
    return dec2

# Decisiones base
dec_codes = np.where(proba_all < thr["t_approve"], 0, np.where(proba_all > thr["t_reject"], 2, 1))

# Aplicar guardrails si corresponde
if use_gr:
    params = {
        "dti_hi": dti_hi, "cs_low": cs_low, "min_months_employed": int(min_mo),
        "income_lo": income_lo, "loan_hi": loan_hi, "rate_hi": rate_hi,
        "risky_purposes": [p for p in risky_purposes if p.strip()]
    }
    dec_codes = apply_guardrails_streamlit(df_holdout, dec_codes, params)

# Recalcular tabla por decisión con (o sin) guardrails
tmp = df_holdout.copy()
tmp["proba"] = proba_all
tmp["decision"] = np.where(dec_codes==0,"Approved", np.where(dec_codes==1,"Review","Rejected"))
by_dec = (
    tmp.groupby("decision").agg(
        rows=("decision","size"),
        rate=("decision", lambda s: len(s)/len(tmp)),
        bad_rate=(TARGET, "mean")
    ).reindex(["Approved","Review","Rejected"])
)
st.subheader("Operación (holdout) por bucket de decisión" + (" con guardrails" if use_gr else ""))
st.dataframe(by_dec.style.format({"rate":"{:.2%}", "bad_rate":"{:.2%}"}))
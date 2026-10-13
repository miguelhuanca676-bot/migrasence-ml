"""
MigraSense ML · Entrenamiento y evaluación honesta.

Uso:
    python entrenar.py                          # solo pacientes simulados (base de estudios)
    python entrenar.py --reales reales.json     # + datos reales exportados desde la BD

Cómo se evalúa (para poder DEMOSTRAR que funciona):
  - Se separan PERSONAS completas: el modelo se evalúa con gente que nunca vio.
  - Se compara contra dos referencias simples:
      a) "solo cuestionario": la probabilidad sale solo de la frecuencia declarada.
      b) un modelo lineal (regresión logística) con las mismas características.
  - Métricas: AUC (0.5 = azar, 1 = perfecto), precisión-recall, Brier (error de
    la probabilidad; menor es mejor) y calibración (si dice 60%, ¿pasa ~60%?).
  - Si hay datos reales suficientes, se evalúa APARTE con ellos.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import caracteristicas as C
import conocimiento as K
import simulador

VERSION = "2.0.0"
DIA, HORA = 86400.0, 3600.0
H = K.HORIZONTE_HORAS


# ---------------------------------------------------------------------------
# De pacientes a filas de entrenamiento
# ---------------------------------------------------------------------------
def muestras_de_paciente(paciente: dict, rng, fin: float, por_dia: float = 0.35) -> list[tuple[dict, int]]:
    """Fotos del paciente en distintos momentos + si tuvo migraña en las 48 h siguientes.
    Solo momentos con 48 h ya transcurridas (si no, todavía no sabemos la respuesta)."""
    prep = C.preparar(paciente)
    tiempos = [x for x in prep["eps_t"] + prep["regs_t"]]
    if not tiempos:
        return []
    inicio = paciente.get("_inicio", min(tiempos))
    limite = fin - H * HORA

    momentos = [r + 60 for r in prep["regs_t"]]                    # justo después de anotar síntomas
    dias = int((limite - inicio) / DIA)
    for d in range(7, max(dias, 7)):
        if rng.random() < por_dia:                                   # momentos cualquiera del día
            momentos.append(inicio + d * DIA + rng.uniform(6, 23) * HORA)
    momentos = [m for m in momentos if inicio + 3 * DIA <= m <= limite]
    return [(C.calcular(prep, m), C.etiqueta(prep, m)) for m in momentos]


def construir(pacientes: list[dict], rng, fin_fijo: float | None = None, grupo0: int = 0, peso: float = 1.0):
    filas, y, grupos, pesos = [], [], [], []
    for i, p in enumerate(pacientes):
        fin = fin_fijo or (p["_inicio"] + p["_dias"] * DIA)
        for f, et in muestras_de_paciente(p, rng, fin):
            filas.append(f)
            y.append(et)
            grupos.append(grupo0 + i)
            pesos.append(peso)
    X = pd.DataFrame(filas, columns=C.columnas())
    return X, np.array(y), np.array(grupos), np.array(pesos)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def prob_solo_cuestionario(X: pd.DataFrame) -> np.ndarray:
    """Referencia: si solo supiéramos la frecuencia declarada (proceso de Poisson)."""
    return 1 - np.exp(-X["ataques_mes_perfil"].to_numpy() / 30.0 * (H / 24.0))


def calibracion(y, p, cortes=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001)) -> list[dict]:
    tabla = []
    for a, b in zip(cortes[:-1], cortes[1:]):
        m = (p >= a) & (p < b)
        if m.sum() >= 30:
            tabla.append({"rango": f"{int(a*100)}-{min(int(b*100),100)}%", "casos": int(m.sum()),
                          "prob_media": round(float(p[m].mean()), 3), "real": round(float(y[m].mean()), 3)})
    return tabla


def metricas(y, p) -> dict:
    return {"casos": int(len(y)), "tasa_real": round(float(y.mean()), 3),
            "auc": round(float(roc_auc_score(y, p)), 3),
            "pr_auc": round(float(average_precision_score(y, p)), 3),
            "brier": round(float(brier_score_loss(y, p)), 4)}


def alerta(y, p, umbral) -> dict:
    m = p >= umbral
    return {"umbral": umbral, "alertas": int(m.sum()),
            "aciertos_%": round(100 * float(y[m].mean()), 1) if m.any() else None,
            "migranas_avisadas_%": round(100 * float(m[y == 1].mean()), 1) if (y == 1).any() else None}


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def nuevo_modelo():
    gb = HistGradientBoostingClassifier(max_iter=350, learning_rate=0.05, max_leaf_nodes=31,
                                        min_samples_leaf=60, l2_regularization=1.0, random_state=7)
    return CalibratedClassifierCV(gb, method="isotonic", cv=3)


def entrenar(n_simulados: int = 1600, reales: list[dict] | None = None, semilla: int = 42,
             ahora_reales: float | None = None) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(semilla)
    sim = simulador.simular_poblacion(n_simulados, semilla)
    X, y, g, w = construir(sim, rng)

    # Datos reales: pesan más (son la verdad de esta app)
    Xr = yr = gr = None
    if reales:
        ahora_reales = ahora_reales or time.time()
        for p in reales:
            ts = [C._ts(e["inicio"]) for e in p.get("episodios", [])] + [C._ts(r["fecha"]) for r in p.get("registros", [])]
            p["_inicio"] = min(ts) if ts else ahora_reales
        Xr, yr, gr, wr = construir(reales, rng, fin_fijo=ahora_reales, grupo0=10**6, peso=3.0)

    # Evaluación con personas que el modelo nunca vio
    sep = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=semilla)
    itr, ite = next(sep.split(X, y, g))
    Xtr, ytr, wtr = X.iloc[itr], y[itr], w[itr]
    if Xr is not None and len(Xr):
        Xtr = pd.concat([Xtr, Xr]); ytr = np.concatenate([ytr, yr]); wtr = np.concatenate([wtr, np.full(len(yr), 3.0)])

    modelo = nuevo_modelo().fit(Xtr, ytr, sample_weight=wtr)
    lineal = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X.iloc[itr], y[itr])

    Xte, yte = X.iloc[ite], y[ite]
    p_mod = modelo.predict_proba(Xte)[:, 1]
    p_lin = lineal.predict_proba(Xte)[:, 1]
    p_base = prob_solo_cuestionario(Xte)

    tras_aviso = (Xte["horas_desde_ultimo_registro"] <= 1).to_numpy()
    reporte = {
        "version": VERSION,
        "entrenado": datetime.now().isoformat(timespec="seconds"),
        "horizonte_horas": H,
        "datos": {"pacientes_simulados": n_simulados, "filas_simuladas": int(len(X)),
                  "pacientes_reales": len(reales or []), "filas_reales": int(len(Xr)) if Xr is not None else 0},
        "prueba_personas_nuevas": {
            "modelo": metricas(yte, p_mod),
            "referencia_solo_cuestionario": metricas(yte, p_base),
            "referencia_lineal": metricas(yte, p_lin),
            "justo_despues_de_anotar_sintomas": metricas(yte[tras_aviso], p_mod[tras_aviso]) if tras_aviso.sum() > 50 else None,
            "calibracion": calibracion(yte, p_mod),
            "alertas": [alerta(yte, p_mod, u) for u in (0.5, 0.6, 0.7)],
        },
    }

    # Qué mira más el modelo (importancia por permutación sobre una muestra)
    m = rng.choice(len(Xte), size=min(6000, len(Xte)), replace=False)
    imp = permutation_importance(modelo, Xte.iloc[m], yte[m], scoring="roc_auc", n_repeats=3, random_state=1)
    orden = np.argsort(-imp.importances_mean)[:12]
    reporte["lo_que_mas_pesa"] = [{"caracteristica": Xte.columns[i], "importancia": round(float(imp.importances_mean[i]), 4)}
                                  for i in orden]

    # Evaluación aparte con datos reales (si hay suficientes)
    if Xr is not None and len(Xr) >= 200 and len(set(gr)) >= 5 and 0 < yr.sum() < len(yr):
        sep_r = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=semilla)
        a, b = next(sep_r.split(Xr, yr, gr))
        solo_sim = nuevo_modelo().fit(X, y)
        reporte["prueba_datos_reales"] = {
            "modelo_base_estudios": metricas(yr[b], solo_sim.predict_proba(Xr.iloc[b])[:, 1]),
            "referencia_solo_cuestionario": metricas(yr[b], prob_solo_cuestionario(Xr.iloc[b])),
        }
    else:
        reporte["prueba_datos_reales"] = "Aún no hay suficientes datos reales (mínimo 5 personas y 200 momentos con 48 h de seguimiento)."

    # Modelo final: TODOS los datos (la evaluación de arriba ya se hizo aparte)
    Xall, yall, wall = X, y, w
    if Xr is not None and len(Xr):
        Xall = pd.concat([X, Xr]); yall = np.concatenate([y, yr]); wall = np.concatenate([w, np.full(len(yr), 3.0)])
    final = nuevo_modelo().fit(Xall, yall, sample_weight=wall)
    reporte["segundos"] = round(time.time() - t0, 1)
    return {"modelo": final, "columnas": C.columnas(), "version": VERSION, "reporte": reporte}


def guardar(paquete: dict, ruta: str = "modelo_migrasense.joblib", ruta_reporte: str = "reporte_modelo.json"):
    joblib.dump(paquete, ruta, compress=3)
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        json.dump(paquete["reporte"], f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulados", type=int, default=1600)
    ap.add_argument("--reales", help="JSON con pacientes reales exportados desde la BD")
    a = ap.parse_args()
    reales = json.load(open(a.reales, encoding="utf-8"))["pacientes"] if a.reales else None
    paquete = entrenar(a.simulados, reales)
    guardar(paquete)
    print(json.dumps(paquete["reporte"], ensure_ascii=False, indent=2))

"""
MigraSense ML · Simulador de pacientes basado en estudios (conocimiento.py).

Crea personas ficticias con perfil, días de vida y migrañas, con el mismo formato
de datos que guarda la app. Sirve para que el modelo "aprenda a pensar" antes de
tener usuarios reales.

Qué reproduce:
  - Frecuencia de ataques según el cuestionario (con variación entre personas).
  - Personas "regulares" (ataques casi cada X días) y personas "irregulares".
  - Unos 2 días de pausa después de cada ataque.
  - Ciclo menstrual en mujeres con componente hormonal [MACGREGOR].
  - Rachas de estrés con el efecto "bajada de estrés" [LIPTON].
  - Noches malas y cafeína alta como multiplicadores [KELMAN, MOSTOFSKY].
  - Síntomas premonitorios 2-48 h antes, con los síntomas típicos y su
    momento [ICHD3, GIFFIN, REVISION], y falsas alarmas.
  - Que la gente no anota todo (cumplimiento < 100%).
Lo que NO ve el modelo: rachas de estrés, ciclo menstrual, noches malas. Igual
que en la vida real, eso queda como "ruido" que limita la precisión posible.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

import conocimiento as K

DIA = 86400.0
HORA = 3600.0
INICIO = datetime(2026, 1, 1).timestamp()


def _elegir(rng, opciones: dict):
    claves = list(opciones)
    p = np.array([opciones[c] for c in claves], dtype=float)
    return claves[rng.choice(len(claves), p=p / p.sum())]


def perfil_aleatorio(rng) -> dict:
    genero = _elegir(rng, {"femenino": 0.72, "masculino": 0.26, "otro": 0.02})  # migraña ~3:1 mujeres
    cafeina = _elegir(rng, {"si": 0.7, "no": 0.3})
    condiciones = []
    if genero == "femenino" and rng.random() < 0.45:
        condiciones.append("hormonal")
    if rng.random() < 0.25:
        condiciones.append("trastornos_sueno")
    if rng.random() < 0.30:
        condiciones.append("ansiedad_depresion")
    return {
        "frecuencia": _elegir(rng, {"menos_1_mes": 0.18, "1_3_mes": 0.37, "1_2_semana": 0.28,
                                    "3_mas_semana": 0.12, "casi_diario": 0.05}),
        "sueno": _elegir(rng, {"menos_5": 0.15, "5_6": 0.40, "7_8": 0.38, "mas_8": 0.07}),
        "estres": _elegir(rng, {"bajo": 0.12, "moderado": 0.38, "alto": 0.33, "muy_alto": 0.17}),
        "cafeina": cafeina,
        "porcionesCafeina": _elegir(rng, {"1": 0.4, "2_3": 0.45, "4_mas": 0.15}) if cafeina == "si" else None,
        "tiempoMigrana": _elegir(rng, {"menos_1_anio": 0.1, "1_3_anios": 0.25,
                                       "4_10_anios": 0.35, "mas_10_anios": 0.3}),
        "genero": genero,
        "condiciones": condiciones or ["ninguna"],
    }


def _sintomas(rng, probs: dict, escala: float) -> list[str]:
    elegidos = [s for s, p in probs.items() if rng.random() < p * escala]
    if not elegidos:
        s = list(probs)
        w = np.array([probs[x] for x in s])
        elegidos = [s[rng.choice(len(s), p=w / w.sum())]]
    return elegidos


def simular_paciente(rng, dias: int | None = None) -> dict:
    perfil = perfil_aleatorio(rng)
    if rng.random() < 0.08:                      # algunos no completan el cuestionario
        perfil_visible = None
    else:
        perfil_visible = perfil
    dias = dias or int(rng.integers(90, 300))

    # Tasa real de esta persona: lo que declara, con error (la gente estima mal)
    tasa_mes = K.ATAQUES_MES[perfil["frecuencia"]] * float(rng.lognormal(0, 0.35))
    tasa_mes *= K.RIESGO_SUENO[perfil["sueno"]] / 1.15 * K.RIESGO_ESTRES[perfil["estres"]] / 1.1
    if perfil.get("porcionesCafeina") == "4_mas":
        tasa_mes *= K.RIESGO_CAFEINA_ALTA
    base_dia = min(tasa_mes / 30.0, 0.85)

    regularidad = float(rng.uniform(1.0, 4.0))   # 1 = al azar, 4 = muy regular
    intervalo_tipico = 30.0 / max(tasa_mes, 0.3)

    # Desencadenantes que le afectan a esta persona [KELMAN]
    sus_desenc = [d for d, p in K.PREVALENCIA_DESENCADENANTE.items()
                  if (d != "hormonal" or perfil["genero"] == "femenino") and rng.random() < p]
    hormonal = perfil["genero"] == "femenino" and ("hormonal" in perfil["condiciones"] or "hormonal" in sus_desenc)
    sensible_estres = "estres" in sus_desenc
    sensible_sueno = "sueno" in sus_desenc or "trastornos_sueno" in perfil["condiciones"]

    p_aviso = float(rng.uniform(*K.PROB_AVISO_POR_ATAQUE))
    cumple_reg = float(rng.uniform(*K.CUMPLIMIENTO_REGISTRO))
    cumple_ep = float(rng.uniform(*K.CUMPLIMIENTO_EPISODIO))
    falsas_mes = float(rng.uniform(*K.FALSAS_ALARMAS_MES))

    # --- Factores ocultos día a día ---
    ciclo_inicio = float(rng.uniform(0, 28))
    ciclo_largo = float(rng.normal(28, 2))
    estres_dia = np.zeros(dias, dtype=bool)
    d = 0
    while d < dias:
        if rng.random() < 0.05:                  # empieza una racha de estrés
            largo = int(rng.integers(2, 7))
            estres_dia[d:d + largo] = True
            d += largo
        d += 1
    p_mala_noche = 0.18 if perfil["sueno"] in ("menos_5", "5_6") else 0.08

    episodios, registros = [], []
    ultimo = -999.0
    for dia in range(dias):
        h = base_dia
        dsl = dia - ultimo
        if dsl < 2:
            h *= 0.15                            # pausa tras un ataque
        elif ultimo > 0:
            h *= float(np.clip((dsl / intervalo_tipico) ** (regularidad - 1), 0.25, 3.0))
        if hormonal:
            dc = (dia - ciclo_inicio) % ciclo_largo
            if dc >= ciclo_largo - 2:
                h *= K.RIESGO_MENSTRUAL["dias_-2_-1"]
            elif dc < 5:
                h *= K.RIESGO_MENSTRUAL["dias_1_5"]
        if sensible_estres and dia > 0 and estres_dia[dia - 1] and not estres_dia[dia]:
            h *= 1 + (K.RIESGO_BAJADA_ESTRES_6H - 1) * 6 / 24     # 5x durante 6 h ≈ 2x ese día
        if sensible_sueno and rng.random() < p_mala_noche:
            h *= 1.6                                              # [SUPUESTO]
        if rng.random() >= min(h, 0.95):
            continue

        # Hay ataque este día: hora de inicio (más frecuente de madrugada/mañana)
        hora = rng.uniform(4, 10) if rng.random() < 0.45 else rng.uniform(0, 24)
        t_ataque = INICIO + dia * DIA + hora * HORA
        ultimo = dia

        if rng.random() < cumple_ep:
            desenc = [x for x in sus_desenc if rng.random() < 0.35]
            if sensible_estres and dia > 0 and estres_dia[dia - 1] and "estres" not in desenc:
                desenc.append("estres")
            episodios.append({
                "inicio": t_ataque,
                "intensidad": int(np.clip(rng.normal(6.5, 1.8), 1, 10)),
                "duracion": _elegir(rng, {"menos_4h": 0.25, "4_24h": 0.45, "1_3d": 0.25, "mas_3d": 0.05}),
                "desencadenantes": desenc,
            })

        if rng.random() < p_aviso:
            sint = _sintomas(rng, K.P_SINTOMA_EN_AVISO, 0.75)
            if set(sint) & K.SINTOMAS_TEMPRANOS and rng.random() < 0.6:
                antes = rng.uniform(20, 40)
            else:
                antes = float(np.clip(rng.lognormal(np.log(K.HORAS_AVISO["mediana"]), 0.6),
                                      K.HORAS_AVISO["min"], K.HORAS_AVISO["max"]))
            if rng.random() < cumple_reg:
                registros.append({"fecha": t_ataque - antes * HORA,
                                  "intensidad": int(rng.integers(*K.INTENSIDAD_AVISO)),
                                  "sintomas": sint})

    # Falsas alarmas: síntomas que NO terminan en migraña
    n_falsas = rng.poisson(falsas_mes * dias / 30.0)
    for _ in range(n_falsas):
        t = INICIO + rng.uniform(0, dias) * DIA
        registros.append({"fecha": t, "intensidad": int(rng.integers(*K.INTENSIDAD_FALSA)),
                          "sintomas": _sintomas(rng, K.P_SINTOMA_EN_FALSA_ALARMA, 0.75)})

    return {"perfil": perfil_visible, "episodios": episodios, "registros": registros,
            "_dias": dias, "_inicio": INICIO}


def simular_poblacion(n: int, semilla: int = 42) -> list[dict]:
    rng = np.random.default_rng(semilla)
    return [simular_paciente(rng) for _ in range(n)]

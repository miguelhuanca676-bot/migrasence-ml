"""
MigraSense ML · De los registros de una persona a los números que ve el modelo.

ESTE ARCHIVO ES EL ÚNICO que convierte datos en "características". Lo usan:
  - el entrenamiento (pacientes simulados y reales de la BD),
  - la predicción en vivo (lo que manda Laravel desde la BD).
Así el modelo aprende y predice con exactamente las mismas reglas.

Formato de entrada (el mismo que envía Laravel):
{
  "perfil": {                       # GET /api/perfil (puede ser null)
     "frecuencia": "1_3_mes", "sueno": "5_6", "estres": "alto",
     "cafeina": "si", "porcionesCafeina": "2_3", "tiempoMigrana": "4_10_anios",
     "genero": "femenino", "condiciones": ["hormonal"]
  },
  "episodios": [ {"inicio": "2026-10-05T19:00:00", "intensidad": 8,
                  "duracion": "1_3d", "desencadenantes": ["estres"]} ],
  "registros": [ {"fecha": "2026-10-05T09:00:00", "intensidad": 4,
                  "sintomas": ["bostezos_repetidos", "cansancio"]} ]
}
"""
from __future__ import annotations

import bisect
from datetime import datetime

import numpy as np

import conocimiento as K

HORA = 3600.0
DIA = 86400.0
H = K.HORIZONTE_HORAS


def _ts(valor) -> float:
    """Fecha ISO (sin zona = hora local de Bolivia) o número -> segundos."""
    if isinstance(valor, (int, float)):
        return float(valor)
    return datetime.fromisoformat(str(valor).replace("Z", "")).timestamp()


def preparar(paciente: dict) -> dict:
    """Ordena y convierte una vez los eventos de la persona (para calcular rápido muchas veces)."""
    eps = sorted(
        ({"t": _ts(e["inicio"]), "desenc": set(e.get("desencadenantes") or [])}
         for e in (paciente.get("episodios") or [])),
        key=lambda e: e["t"],
    )
    regs = sorted(
        ({"t": _ts(r["fecha"]), "int": int(r.get("intensidad") or 0),
          "sint": set(r.get("sintomas") or [])}
         for r in (paciente.get("registros") or [])),
        key=lambda r: r["t"],
    )
    return {
        "perfil": paciente.get("perfil") or {},
        "eps": eps, "eps_t": [e["t"] for e in eps],
        "regs": regs, "regs_t": [r["t"] for r in regs],
    }


def _perfil(p: dict) -> dict:
    cafeina = p.get("porcionesCafeina") if p.get("cafeina") == "si" else None
    cond = set(p.get("condiciones") or [])
    return {
        "tiene_perfil": 1.0 if p.get("frecuencia") else 0.0,
        "ataques_mes_perfil": K.ATAQUES_MES.get(p.get("frecuencia"), 2.0),
        "sueno_ord": K.ORDEN_SUENO.get(p.get("sueno"), 2),
        "estres_ord": K.ORDEN_ESTRES.get(p.get("estres"), 1),
        "cafeina_ord": K.ORDEN_CAFEINA.get(cafeina, 0),
        "antiguedad_ord": K.ORDEN_ANTIGUEDAD.get(p.get("tiempoMigrana"), 1),
        "mujer": 1.0 if p.get("genero") == "femenino" else 0.0,
        "cond_hormonal": 1.0 if "hormonal" in cond else 0.0,
        "cond_sueno": 1.0 if "trastornos_sueno" in cond else 0.0,
        "cond_ansiedad": 1.0 if "ansiedad_depresion" in cond else 0.0,
    }


def calcular(prep: dict, t: float) -> dict:
    """Características en el instante t usando SOLO lo registrado hasta t (nunca el futuro)."""
    f = _perfil(prep["perfil"])

    # ---------------- Historial de episodios ----------------
    n_ep = bisect.bisect_right(prep["eps_t"], t)
    eps_t = prep["eps_t"][:n_ep]
    eps = prep["eps"][:n_ep]
    primero = min([x for x in (eps_t[:1] + prep["regs_t"][:1]) if x <= t], default=t)
    dias_obs = max((t - primero) / DIA, 0.0)

    f["episodios_total"] = float(n_ep)
    f["episodios_30d"] = float(n_ep - bisect.bisect_left(eps_t, t - 30 * DIA))
    f["episodios_90d"] = float(n_ep - bisect.bisect_left(eps_t, t - 90 * DIA))
    f["dias_desde_ultimo_episodio"] = min((t - eps_t[-1]) / DIA, 120.0) if eps_t else 120.0
    f["dias_observados"] = min(dias_obs, 365.0)

    # Tasa personal "prudente": mezcla lo que dice el perfil con lo observado
    # (con pocos días manda el perfil; con muchos, manda el historial real).
    meses = min(dias_obs, 365.0) / 30.0
    k = 1.0
    f["tasa_mensual_personal"] = (f["episodios_90d" if dias_obs >= 90 else "episodios_total"]
                                  + k * f["ataques_mes_perfil"]) / (min(meses, 3.0) + k)

    if n_ep >= 3:
        gaps = np.diff(eps_t[-12:]) / DIA
        gaps = gaps[gaps >= 0.5]
        intervalo = float(np.median(gaps)) if len(gaps) else 0.0
    else:
        intervalo = 0.0
    f["intervalo_medio_dias"] = intervalo
    f["ciclo_relativo"] = min(f["dias_desde_ultimo_episodio"] / intervalo, 5.0) if intervalo > 0 else -1.0

    for d in ("estres", "sueno", "hormonal"):
        f[f"frac_desenc_{d}"] = (sum(1 for e in eps if d in e["desenc"]) / n_ep) if n_ep else 0.0

    # ---------------- ¿Sus avisos anteriores acertaron? ----------------
    # Solo avisos con al menos 48 h de antigüedad (si no, aún no sabemos si acertaron).
    n_reg = bisect.bisect_right(prep["regs_t"], t)
    limite = bisect.bisect_right(prep["regs_t"], t - H * HORA)
    aciertos = 0
    for r in prep["regs"][:limite]:
        j = bisect.bisect_right(prep["eps_t"], r["t"])
        if j < len(prep["eps_t"]) and prep["eps_t"][j] - r["t"] <= H * HORA:
            aciertos += 1
    f["avisos_evaluables"] = float(limite)
    # Precisión personal con "freno": empieza en 0.6 (cerca de GIFFIN) y se ajusta con su historia
    f["precision_avisos_personal"] = (aciertos + 0.6 * 3) / (limite + 3)

    # ---------------- Lo que pasa AHORA (últimas horas) ----------------
    regs_t = prep["regs_t"][:n_reg]
    i24 = bisect.bisect_left(regs_t, t - 24 * HORA)
    i48 = bisect.bisect_left(regs_t, t - 48 * HORA)
    recientes = prep["regs"][i48:n_reg]
    f["registros_24h"] = float(n_reg - i24)
    f["registros_48h"] = float(n_reg - i48)
    f["horas_desde_ultimo_registro"] = min((t - regs_t[-1]) / HORA, 96.0) if regs_t else 96.0
    f["intensidad_max_48h"] = float(max((r["int"] for r in recientes), default=0))
    sint48 = set().union(*(r["sint"] for r in recientes)) if recientes else set()
    f["sintomas_distintos_48h"] = float(len(sint48))
    for s in K.SINTOMAS:
        f[f"s_{s}"] = 1.0 if s in sint48 else 0.0

    # ---------------- Momento ----------------
    d = datetime.fromtimestamp(t)
    f["hora_del_dia"] = float(d.hour)
    f["dia_semana"] = float(d.weekday())
    return f


def etiqueta(prep: dict, t: float) -> int:
    """1 si empieza un episodio en las próximas 48 h (lo que el modelo aprende a predecir)."""
    j = bisect.bisect_right(prep["eps_t"], t)
    return int(j < len(prep["eps_t"]) and prep["eps_t"][j] <= t + H * HORA)


def columnas() -> list[str]:
    """Orden fijo de las características (el modelo guardado depende de este orden)."""
    prep = preparar({"perfil": {}, "episodios": [], "registros": []})
    return list(calcular(prep, 1.7e9).keys())

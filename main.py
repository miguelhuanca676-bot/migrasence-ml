"""
MigraSense ML · API de predicción (FastAPI, desplegada en Render).

Endpoints:
  GET  /salud               público  -> ¿está vivo? versión y calidad del modelo
  GET  /metricas            público  -> reporte completo de la evaluación (para la demo)
  POST /predecir            con clave -> probabilidad de migraña en las próximas 48 h + por qué
  POST /reentrenar          con clave -> reentrena en segundo plano con datos reales de la BD
  GET  /reentrenar/estado   con clave -> cómo va el reentrenamiento (y si el nuevo modelo se aceptó)
  GET  /modelo/descargar    con clave -> el modelo activo (Laravel lo guarda en Aiven: no se pierde)
  POST /modelo/cargar       con clave -> Laravel devuelve el modelo aprendido cuando Render reinicia

Aprendizaje continuo (sin perder lo aprendido):
  Render gratis borra el disco al reiniciar. Por eso el modelo aprendido se guarda en la
  base de datos (tabla modelo_ml, a través de Laravel) y Laravel lo vuelve a cargar aquí si
  nota que este servicio arrancó con el modelo original. Un modelo nuevo SOLO se activa si
  pasa el filtro de calidad (no acierta peor que el anterior).

Seguridad:
  - /predecir y /reentrenar exigen la cabecera X-API-Key = variable de entorno ML_API_KEY.
    Si ML_API_KEY no está configurada, esos endpoints NO funcionan (falla cerrado).
  - Este servicio NO se conecta a la base de datos: Laravel le manda los datos de una
    persona ya filtrados (sin nombre, sin correo). Así las claves de Aiven viven solo en Laravel.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import io
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import joblib
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field

import caracteristicas as C
import conocimiento as K

RUTA_MODELO = os.getenv("RUTA_MODELO", "modelo_migrasense.joblib")
RUTA_REPORTE = os.getenv("RUTA_REPORTE", "reporte_modelo.json")
MAX_MODELO_BYTES = 40 * 1024 * 1024
MAX_EVENTOS = 5000

app = FastAPI(title="MigraSense ML", version="2.0.0",
              description="Probabilidad de que empiece una migraña en las próximas 48 horas.")

_estado = {"paquete": None, "cargado": None, "reentreno": {"estado": "sin_iniciar"}}
_candado = threading.Lock()


def _cargar():
    _estado["paquete"] = joblib.load(RUTA_MODELO)
    _estado["cargado"] = datetime.now(timezone.utc).isoformat(timespec="seconds")


try:
    _cargar()
except FileNotFoundError:
    pass   # /salud lo informa; el Dockerfile entrena el modelo al construir la imagen


def _verificar_clave(clave: Optional[str]):
    esperada = os.getenv("ML_API_KEY", "")
    if not esperada:
        raise HTTPException(503, "ML_API_KEY no está configurada en el servidor")
    if not clave or not hmac.compare_digest(clave, esperada):
        raise HTTPException(401, "Clave inválida")


def _ahora_bolivia() -> datetime:
    """Bolivia es UTC-4 todo el año (sin horario de verano)."""
    return (datetime.now(timezone.utc) - timedelta(hours=4)).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Formato de entrada (igual al que arma Laravel desde la BD)
# ---------------------------------------------------------------------------
class Perfil(BaseModel):
    frecuencia: Optional[str] = None
    sueno: Optional[str] = None
    estres: Optional[str] = None
    cafeina: Optional[str] = None
    porcionesCafeina: Optional[str] = None
    tiempoMigrana: Optional[str] = None
    genero: Optional[str] = None
    condiciones: list[str] = []


class Episodio(BaseModel):
    inicio: str
    intensidad: int = Field(5, ge=1, le=10)
    duracion: Optional[str] = None
    desencadenantes: list[str] = []


class Registro(BaseModel):
    fecha: str
    intensidad: int = Field(3, ge=0, le=10)
    sintomas: list[str] = []


class Paciente(BaseModel):
    ahora: Optional[str] = Field(None, description="Hora local de Bolivia, ej. 2026-10-13T16:30:00")
    perfil: Optional[Perfil] = None
    episodios: list[Episodio] = Field(default_factory=list, max_length=MAX_EVENTOS)
    registros: list[Registro] = Field(default_factory=list, max_length=MAX_EVENTOS)


class LoteReal(BaseModel):
    ahora: Optional[str] = None
    pacientes: list[Paciente] = Field(default_factory=list, max_length=20000)
    simulados: int = Field(200, ge=200, le=3000)


# ---------------------------------------------------------------------------
# Explicación en lenguaje simple
# ---------------------------------------------------------------------------
NOMBRE_SINTOMA = {
    "hiperactividad": "hiperactividad", "hipoactividad": "hipoactividad", "depresion": "desánimo",
    "avidez_alimentos": "antojos de comida", "bostezos_repetidos": "bostezos repetidos",
    "astenia": "debilidad", "cervical": "rigidez de cuello", "cansancio": "cansancio",
    "animo_exaltado": "ánimo exaltado", "hambre_atipica": "hambre atípica",
    "concentracion": "dificultad para concentrarte", "luz": "sensibilidad a la luz",
    "ruido": "sensibilidad al ruido", "nauseas": "náuseas", "vision": "visión borrosa",
    "bostezos_palidez": "bostezos o palidez",
}

GRUPOS = {
    "sintomas": ["registros_24h", "registros_48h", "horas_desde_ultimo_registro", "intensidad_max_48h",
                 "sintomas_distintos_48h"] + [f"s_{s}" for s in K.SINTOMAS],
    "ciclo": ["dias_desde_ultimo_episodio", "ciclo_relativo"],
    "historial": ["episodios_30d", "episodios_90d", "tasa_mensual_personal", "episodios_total"],
    "perfil": ["sueno_ord", "estres_ord", "cafeina_ord", "cond_hormonal", "cond_sueno", "cond_ansiedad"],
}
NEUTRO = {"registros_24h": 0, "registros_48h": 0, "horas_desde_ultimo_registro": 96, "intensidad_max_48h": 0,
          "sintomas_distintos_48h": 0, "sueno_ord": 2, "estres_ord": 1, "cafeina_ord": 1,
          "cond_hormonal": 0, "cond_sueno": 0, "cond_ansiedad": 0}


def _prob(modelo, filas: list[dict], columnas: list[str]):
    return modelo.predict_proba(pd.DataFrame(filas, columns=columnas))[:, 1]


def _explicar(modelo, f: dict, columnas: list[str], p: float) -> list[dict]:
    """Cuánto cambia la probabilidad si "apagamos" cada grupo de señales."""
    variantes = []
    for grupo, cols in GRUPOS.items():
        g = dict(f)
        for c in cols:
            if c.startswith("s_"):
                g[c] = 0.0
            elif c in NEUTRO:
                g[c] = NEUTRO[c]
            elif grupo == "ciclo":
                g["dias_desde_ultimo_episodio"] = f["intervalo_medio_dias"] * 0.5 if f["intervalo_medio_dias"] else 15
                g["ciclo_relativo"] = 0.5 if f["intervalo_medio_dias"] else -1
            elif grupo == "historial":
                g["episodios_30d"] = g["episodios_90d"] = g["episodios_total"] = 0
                g["tasa_mensual_personal"] = f["ataques_mes_perfil"]
        variantes.append(g)
    sin = _prob(modelo, variantes, columnas)

    textos = []
    for (grupo, _), q in zip(GRUPOS.items(), sin):
        efecto = round(float(p - q) * 100, 1)       # puntos porcentuales que aporta este grupo
        if abs(efecto) < 3:
            continue
        sube = efecto > 0
        if grupo == "sintomas":
            presentes = [NOMBRE_SINTOMA[s] for s in K.SINTOMAS if f[f"s_{s}"]]
            txt = (f"Anotaste síntomas de aviso en las últimas 48 h ({', '.join(presentes[:4])})"
                   if sube and presentes else
                   "No anotaste síntomas de aviso recientemente" if not sube else "Tus registros recientes")
        elif grupo == "ciclo":
            dias_u = int(f["dias_desde_ultimo_episodio"])
            ritmo = f" y tu ritmo habitual es de unos {int(f['intervalo_medio_dias'])} días" if f["intervalo_medio_dias"] else ""
            txt = (f"Han pasado {dias_u} días desde tu último episodio{ritmo}" if sube else
                   f"Tu último episodio fue hace {dias_u} día(s){ritmo}: todavía no toca según tu ritmo")
        elif grupo == "historial":
            n30 = int(f["episodios_30d"])
            txt = (f"Tuviste {n30} episodio(s) en los últimos 30 días" if sube else
                   f"Tus episodios recientes ({n30} en 30 días) no muestran un aumento")
        else:
            txt = "Tu perfil (sueño, estrés, cafeína y condiciones)"
        textos.append({"factor": grupo, "texto": txt, "efecto_puntos": efecto,
                       "direccion": "sube" if sube else "baja"})
    if not textos:
        textos.append({"factor": "general", "efecto_puntos": 0.0, "direccion": "baja" if p < 0.25 else "sube",
                       "texto": "No hay señales recientes que cambien mucho tu riesgo; la predicción sale de tu historial y tu perfil"})
    return sorted(textos, key=lambda x: -abs(x["efecto_puntos"]))


def _nivel(p: float) -> tuple[str, str]:
    if p >= 0.70:
        return "muy_alto", "Probabilidad muy alta de migraña en las próximas 48 h. Descansa, hidrátate, evita tus desencadenantes y ten tu medicación a mano."
    if p >= 0.50:
        return "alto", "Probabilidad alta en las próximas 48 h. Cuida el sueño, la comida y el estrés."
    if p >= 0.25:
        return "moderado", "Probabilidad moderada. Sigue anotando tus síntomas para afinar la predicción."
    return "bajo", "Probabilidad baja en las próximas 48 h."


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/salud")
def salud():
    paq = _estado["paquete"]
    if not paq:
        return {"ok": False, "modelo_cargado": False}
    rep = paq["reporte"]
    return {"ok": True, "modelo_cargado": True, "version": paq["version"], "entrenado": rep["entrenado"],
            "auc_personas_nuevas": rep["prueba_personas_nuevas"]["modelo"]["auc"],
            "pacientes_reales_usados": rep["datos"]["pacientes_reales"],
            "aprendido_con_usuarios": "-r" in str(paq["version"]),   # False = modelo original de la imagen
            "reentrenamiento": _estado["reentreno"]["estado"]}


@app.get("/metricas")
def metricas():
    if not _estado["paquete"]:
        raise HTTPException(503, "Modelo no cargado")
    return _estado["paquete"]["reporte"]


@app.post("/predecir")
def predecir(datos: Paciente, x_api_key: Optional[str] = Header(None)):
    _verificar_clave(x_api_key)
    paq = _estado["paquete"]
    if not paq:
        raise HTTPException(503, "Modelo no cargado")

    ahora = datos.ahora or _ahora_bolivia().isoformat(timespec="seconds")
    try:
        t = C._ts(ahora)
        prep = C.preparar(datos.model_dump())
    except (ValueError, TypeError) as e:
        raise HTTPException(422, f"Fecha inválida: {e}")

    f = C.calcular(prep, t)
    p = float(_prob(paq["modelo"], [f], paq["columnas"])[0])
    nivel, mensaje = _nivel(p)

    # ¿Cuánto podemos confiar? Depende de cuánto historial tiene la persona
    dias, n_ep = f["dias_observados"], f["episodios_total"]
    confianza = "alta" if dias >= 60 and n_ep >= 4 else "media" if dias >= 14 and n_ep >= 1 else "baja"

    return {
        "ok": True,
        "probabilidad": round(p, 4),
        "porcentaje": round(p * 100, 1),
        "nivel": nivel,
        "mensaje": mensaje,
        "ventana_horas": K.HORIZONTE_HORAS,
        "confianza": confianza,
        "factores": _explicar(paq["modelo"], f, paq["columnas"], p),
        "datos_usados": {"episodios": len(datos.episodios), "registros": len(datos.registros),
                         "dias_de_historial": round(dias, 1), "tiene_perfil": bool(f["tiene_perfil"])},
        "modelo_version": paq["version"],
        "calculado_para": ahora,
    }


TOLERANCIA_AUC = 0.01      # variación normal entre entrenamientos
AUC_MINIMO = 0.75          # nunca se activa un modelo por debajo de esto


def _auc(paq: dict | None) -> float | None:
    try:
        return float(paq["reporte"]["prueba_personas_nuevas"]["modelo"]["auc"])
    except (TypeError, KeyError, ValueError):
        return None


def filtro_calidad(nuevo: dict, actual: dict | None) -> tuple[bool, str]:
    """¿Se puede activar el modelo nuevo? Solo si no acierta peor que el que está funcionando."""
    auc_n, auc_a = _auc(nuevo), _auc(actual)
    if auc_n is None:
        return False, "El modelo nuevo no tiene evaluación"
    if auc_n < AUC_MINIMO:
        return False, f"AUC {auc_n:.3f} por debajo del mínimo {AUC_MINIMO}"
    if auc_a is not None and auc_n < auc_a - TOLERANCIA_AUC:
        return False, f"Acierta peor que el actual (AUC {auc_n:.3f} < {auc_a:.3f})"
    real = nuevo["reporte"].get("prueba_datos_reales")
    if isinstance(real, dict) and "modelo_con_datos_reales" in real:
        con, base = real["modelo_con_datos_reales"]["auc"], real["modelo_base_estudios"]["auc"]
        if con < base - 0.02:
            return False, f"Con personas reales acierta peor que el modelo base (AUC {con:.3f} < {base:.3f})"
    return True, (f"Aceptado: AUC {auc_n:.3f}" + (f" (antes {auc_a:.3f})" if auc_a is not None else ""))


def _a_bytes(paq: dict) -> bytes:
    buf = io.BytesIO()
    joblib.dump(paq, buf, compress=3)
    return buf.getvalue()


def _reentrenar(lote: dict):
    import entrenar
    with _candado:
        _estado["reentreno"] = {"estado": "en_curso", "inicio": time.time()}
        try:
            reales = [p for p in lote["pacientes"] if p.get("episodios") or p.get("registros")]
            ahora = C._ts(lote["ahora"]) if lote.get("ahora") else C._ts(_ahora_bolivia().isoformat())
            version = f"{entrenar.VERSION}-r{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            paq = entrenar.entrenar(n_simulados=lote["simulados"], reales=reales, ahora_reales=ahora,
                                    version=version, rapido=True)
            actual = _estado["paquete"]
            aceptado, motivo = filtro_calidad(paq, actual)
            if aceptado:
                entrenar.guardar(paq, RUTA_MODELO, RUTA_REPORTE)
                _estado["paquete"] = paq
            rep = paq["reporte"]
            _estado["reentreno"] = {"estado": "terminado", "aceptado": aceptado, "motivo": motivo,
                                    "version": version, "version_anterior": actual["version"] if actual else None,
                                    "segundos": rep["segundos"], "pacientes_reales": len(reales),
                                    "filas_reales": rep["datos"]["filas_reales"],
                                    "auc_personas_nuevas": _auc(paq), "auc_anterior": _auc(actual),
                                    "prueba_datos_reales": rep["prueba_datos_reales"]}
        except Exception as e:   # noqa: BLE001 - se informa en /reentrenar/estado
            _estado["reentreno"] = {"estado": "error", "detalle": str(e)[:300]}


@app.post("/reentrenar", status_code=202)
def reentrenar(lote: LoteReal, tareas: BackgroundTasks, x_api_key: Optional[str] = Header(None)):
    _verificar_clave(x_api_key)
    if _estado["reentreno"].get("estado") == "en_curso":
        raise HTTPException(409, "Ya hay un reentrenamiento en curso")
    tareas.add_task(_reentrenar, lote.model_dump())
    return {"ok": True, "mensaje": "Reentrenamiento iniciado. Consulta /reentrenar/estado.",
            "pacientes_recibidos": len(lote.pacientes)}


@app.get("/reentrenar/estado")
def estado_reentreno(x_api_key: Optional[str] = Header(None)):
    _verificar_clave(x_api_key)
    return _estado["reentreno"]


# ---------------------------------------------------------------------------
# Guardar / recuperar lo aprendido (Laravel lo guarda en la base de datos)
# ---------------------------------------------------------------------------
@app.get("/modelo/descargar")
def modelo_descargar(x_api_key: Optional[str] = Header(None)):
    _verificar_clave(x_api_key)
    paq = _estado["paquete"]
    if not paq:
        raise HTTPException(503, "Modelo no cargado")
    datos = _a_bytes(paq)
    return Response(content=datos, media_type="application/octet-stream",
                    headers={"X-Modelo-Version": paq["version"], "X-Sha256": hashlib.sha256(datos).hexdigest()})


@app.post("/modelo/cargar")
async def modelo_cargar(request: Request, x_api_key: Optional[str] = Header(None),
                        x_sha256: Optional[str] = Header(None)):
    """Recibe (en base64) un modelo que ESTE servicio entrenó antes y Laravel guardó en la BD."""
    _verificar_clave(x_api_key)
    cuerpo = await request.body()
    if len(cuerpo) > MAX_MODELO_BYTES * 4 // 3 + 16:
        raise HTTPException(413, "Modelo demasiado grande")
    try:
        datos = base64.b64decode(cuerpo, validate=True)
    except ValueError:
        raise HTTPException(422, "El modelo debe enviarse en base64")
    if not x_sha256 or not hmac.compare_digest(hashlib.sha256(datos).hexdigest(), x_sha256.lower()):
        raise HTTPException(422, "La huella (sha256) no coincide: archivo dañado")
    try:
        paq = joblib.load(io.BytesIO(datos))
        ok = isinstance(paq, dict) and paq.get("columnas") == C.columnas() and hasattr(paq.get("modelo"), "predict_proba")
    except Exception:   # noqa: BLE001
        ok = False
    if not ok:
        raise HTTPException(422, "El archivo no es un modelo de MigraSense compatible")
    with _candado:
        _estado["paquete"] = paq
        _estado["cargado"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            joblib.dump(paq, RUTA_MODELO, compress=3)
        except OSError:
            pass   # en memoria igual funciona
    return {"ok": True, "version": paq["version"], "auc_personas_nuevas": _auc(paq)}

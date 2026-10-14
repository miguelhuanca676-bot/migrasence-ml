"""Pruebas de la API. Ejecutar: pytest -q"""
import os
from datetime import datetime, timedelta

os.environ["ML_API_KEY"] = "clave-de-prueba"

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

cliente = TestClient(main.app)
CLAVE = {"X-API-Key": "clave-de-prueba"}
AHORA = datetime(2026, 10, 13, 16, 0)


def iso(dt):
    return dt.isoformat(timespec="seconds")


def paciente_frecuente_con_aviso():
    """Migraña cada ~6 días, último episodio hace 6 días, y anotó bostezos y luz hace 3 h."""
    eps = [{"inicio": iso(AHORA - timedelta(days=6 * i, hours=2)), "intensidad": 7,
            "duracion": "4_24h", "desencadenantes": ["estres"]} for i in range(1, 12)]
    regs = [{"fecha": iso(e_dt - timedelta(hours=8)), "intensidad": 6, "sintomas": ["bostezos_repetidos", "cansancio"]}
            for e_dt in (AHORA - timedelta(days=6 * i, hours=2) for i in range(1, 12))]
    regs.append({"fecha": iso(AHORA - timedelta(hours=3)), "intensidad": 7,
                 "sintomas": ["bostezos_repetidos", "luz", "concentracion"]})
    return {"ahora": iso(AHORA), "episodios": eps, "registros": regs,
            "perfil": {"frecuencia": "1_2_semana", "sueno": "5_6", "estres": "alto", "cafeina": "si",
                       "porcionesCafeina": "2_3", "tiempoMigrana": "4_10_anios", "genero": "femenino",
                       "condiciones": ["hormonal"]}}


def paciente_tranquilo():
    """Migraña rara (último episodio hace 3 días) y sin síntomas recientes."""
    return {"ahora": iso(AHORA),
            "episodios": [{"inicio": iso(AHORA - timedelta(days=d)), "intensidad": 5, "duracion": "menos_4h"}
                          for d in (3, 50, 95)],
            "registros": [],
            "perfil": {"frecuencia": "menos_1_mes", "sueno": "7_8", "estres": "bajo", "cafeina": "no",
                       "tiempoMigrana": "1_3_anios", "genero": "masculino", "condiciones": ["ninguna"]}}


def test_salud_publica():
    r = cliente.get("/salud").json()
    assert r["ok"] and r["modelo_cargado"] and r["auc_personas_nuevas"] > 0.8


def test_predecir_exige_clave():
    assert cliente.post("/predecir", json=paciente_tranquilo()).status_code == 401
    assert cliente.post("/predecir", json=paciente_tranquilo(), headers={"X-API-Key": "otra"}).status_code == 401


def test_riesgo_alto_vs_bajo_y_explicacion():
    alto = cliente.post("/predecir", json=paciente_frecuente_con_aviso(), headers=CLAVE).json()
    bajo = cliente.post("/predecir", json=paciente_tranquilo(), headers=CLAVE).json()
    assert alto["ok"] and bajo["ok"]
    assert alto["probabilidad"] > 0.5 > bajo["probabilidad"]
    assert alto["nivel"] in ("alto", "muy_alto") and bajo["nivel"] == "bajo"
    assert any(f["factor"] == "sintomas" and f["direccion"] == "sube" for f in alto["factores"])
    assert alto["confianza"] == "alta"


def test_sin_datos_no_falla():
    r = cliente.post("/predecir", json={}, headers=CLAVE).json()
    assert r["ok"] and 0 <= r["probabilidad"] <= 1 and r["confianza"] == "baja"


def test_datos_invalidos():
    malo = paciente_tranquilo()
    malo["episodios"][0]["intensidad"] = 15
    assert cliente.post("/predecir", json=malo, headers=CLAVE).status_code == 422
    malo = paciente_tranquilo()
    malo["ahora"] = "no-es-fecha"
    assert cliente.post("/predecir", json=malo, headers=CLAVE).status_code == 422


def test_no_usa_el_futuro():
    """Un episodio DESPUÉS de 'ahora' no debe cambiar la predicción (no hacer trampa)."""
    base = paciente_tranquilo()
    con_futuro = paciente_tranquilo()
    con_futuro["episodios"].append({"inicio": iso(AHORA + timedelta(hours=10)), "intensidad": 9})
    a = cliente.post("/predecir", json=base, headers=CLAVE).json()["probabilidad"]
    b = cliente.post("/predecir", json=con_futuro, headers=CLAVE).json()["probabilidad"]
    assert a == b


def test_reentrenar_exige_clave():
    assert cliente.post("/reentrenar", json={"pacientes": []}).status_code == 401
    assert cliente.get("/reentrenar/estado").status_code == 401


# ---------------------------------------------------------------------------
# Aprendizaje continuo: filtro de calidad y guardar/recuperar el modelo
# ---------------------------------------------------------------------------
import base64  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402


def _con_auc(paq, auc, version="2.0.0-rprueba", real=None):
    p = dict(paq)
    rep = copy.deepcopy(paq["reporte"])
    rep["prueba_personas_nuevas"]["modelo"]["auc"] = auc
    if real is not None:
        rep["prueba_datos_reales"] = real
    p["reporte"], p["version"] = rep, version
    return p


def test_filtro_calidad_acepta_igual_o_mejor_y_rechaza_peor():
    actual = main._estado["paquete"]
    base = main._auc(actual)
    assert main.filtro_calidad(_con_auc(actual, base + 0.01), actual)[0] is True
    assert main.filtro_calidad(_con_auc(actual, base - 0.005), actual)[0] is True     # variación normal
    ok, motivo = main.filtro_calidad(_con_auc(actual, base - 0.05), actual)
    assert ok is False and "peor" in motivo
    assert main.filtro_calidad(_con_auc(actual, 0.70), None)[0] is False             # bajo el mínimo
    malo_real = {"modelo_con_datos_reales": {"auc": 0.60}, "modelo_base_estudios": {"auc": 0.80}}
    assert main.filtro_calidad(_con_auc(actual, base, real=malo_real), actual)[0] is False


def test_descargar_y_cargar_el_modelo_ida_y_vuelta():
    assert cliente.get("/modelo/descargar").status_code == 401                       # sin clave
    r = cliente.get("/modelo/descargar", headers=CLAVE)
    assert r.status_code == 200 and r.headers["X-Sha256"] == hashlib.sha256(r.content).hexdigest()
    cuerpo = base64.b64encode(r.content)
    ok = cliente.post("/modelo/cargar", content=cuerpo, headers={**CLAVE, "X-Sha256": r.headers["X-Sha256"]})
    assert ok.status_code == 200 and ok.json()["version"] == r.headers["X-Modelo-Version"]


def test_cargar_rechaza_archivos_danados_o_ajenos():
    r = cliente.get("/modelo/descargar", headers=CLAVE)
    danado = cliente.post("/modelo/cargar", content=base64.b64encode(r.content[:-10]),
                          headers={**CLAVE, "X-Sha256": r.headers["X-Sha256"]})
    assert danado.status_code == 422
    import joblib, io  # noqa: E401
    buf = io.BytesIO(); joblib.dump({"hola": 1}, buf); datos = buf.getvalue()
    ajeno = cliente.post("/modelo/cargar", content=base64.b64encode(datos),
                         headers={**CLAVE, "X-Sha256": hashlib.sha256(datos).hexdigest()})
    assert ajeno.status_code == 422
    assert cliente.post("/modelo/cargar", content=b"abc", headers={"X-Sha256": "x"}).status_code == 401


def test_reentrenar_solo_activa_el_modelo_si_pasa_el_filtro(monkeypatch, tmp_path):
    import entrenar
    actual = main._estado["paquete"]
    monkeypatch.setattr(main, "RUTA_MODELO", str(tmp_path / "m.joblib"))
    monkeypatch.setattr(main, "RUTA_REPORTE", str(tmp_path / "r.json"))
    lote = {"ahora": iso(AHORA), "simulados": 200, "pacientes": [paciente_frecuente_con_aviso()]}

    monkeypatch.setattr(entrenar, "entrenar", lambda **k: _con_auc(actual, main._auc(actual) - 0.1, k["version"]))
    main._reentrenar(lote)
    est = cliente.get("/reentrenar/estado", headers=CLAVE).json()
    assert est["estado"] == "terminado" and est["aceptado"] is False
    assert main._estado["paquete"] is actual                                         # sigue el de antes

    monkeypatch.setattr(entrenar, "entrenar", lambda **k: _con_auc(actual, main._auc(actual) + 0.01, k["version"]))
    main._reentrenar(lote)
    est = cliente.get("/reentrenar/estado", headers=CLAVE).json()
    assert est["aceptado"] is True and est["pacientes_reales"] == 1
    assert cliente.get("/salud").json()["version"] == est["version"]
    assert cliente.get("/salud").json()["aprendido_con_usuarios"] is True
    main._estado["paquete"] = actual                                                 # dejar como estaba

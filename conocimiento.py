"""
MigraSense ML · Conocimiento médico de partida ("datos de libros").

Cada número que usa el simulador para crear pacientes de entrenamiento sale de
aquí, con su fuente. Hay dos tipos:
  [ESTUDIO]   = cifra publicada (se cita el estudio).
  [SUPUESTO]  = no hay cifra exacta publicada; valor razonable y conservador
                elegido por el equipo, basado en lo que el estudio describe.

Cuando la app tenga datos reales, el modelo se reentrena con ellos y estos
supuestos pesan cada vez menos.

Fuentes:
  [ICHD3]     Headache Classification Committee of the IHS. ICHD-3 (2018): fase
              premonitoria de 2 a 48 h antes del dolor; migraña crónica = >=15 días
              de cefalea al mes.
  [GIFFIN]    Giffin NJ et al. "Premonitory symptoms in migraine: an electronic
              diary study". Neurology 2003;60:935-40. 97 pacientes con diario
              electrónico: cuando anotaron síntomas premonitorios, 72% tuvo migraña
              en <=72 h al menos la mitad de las veces. Más frecuentes: cansancio
              72%, dificultad de concentración 51%, rigidez de cuello 50%. Los
              bostezos fueron de los más predictivos (84%); cansancio, cuello y
              náuseas de los menos predictivos.
  [REVISION]  "The premonitory phase: a crucial stage in migraine", Neurología
              (Elsevier). Prevalencia de síntomas premonitorios 77-92% si se
              pregunta bien; aparecen en promedio 6-10 h antes; hiperactividad y
              euforia más temprano (25-36 h), cuello, cansancio e irritabilidad
              dentro de las 12 h previas; ~3 síntomas por ataque en promedio.
  [KELMAN]    Kelman L. "The triggers or precipitants of the acute migraine
              attack". Cephalalgia 2007;27:394-402. 1.207 pacientes: estrés 80%,
              hormonales 65% (mujeres), no comer 57%, clima 53%, sueño 50%,
              olores 44%, luces 38%, alcohol 38%, alimentos 27%, ejercicio 22%.
  [MACGREGOR] MacGregor EA et al. Neurology 2004 (155 mujeres, diarios): 2 días
              antes de la menstruación +71% de probabilidad; del día 1 al 5, más
              del doble.
  [LIPTON]    Lipton RB et al. "Reduction in perceived stress as a migraine
              trigger". Neurology 2014 (17 pacientes, 2.011 registros): tras una
              bajada del estrés el riesgo fue casi 5 veces mayor en las 6 h
              siguientes.
  [MOSTOFSKY] Mostofsky E et al. Am J Med 2019 (98 adultos, diario 6 semanas):
              3 o más bebidas con cafeína al día se asocian con más migraña ese
              día o el siguiente; 1-2 no.
"""

# ---------------------------------------------------------------------------
# Códigos EXACTOS de la app (js/app.js y tablas sintoma_prodromico / desencadenante)
# ---------------------------------------------------------------------------
SINTOMAS = [
    "hiperactividad", "hipoactividad", "depresion", "avidez_alimentos",
    "bostezos_repetidos", "astenia", "cervical", "cansancio", "animo_exaltado",
    "hambre_atipica", "concentracion", "luz", "ruido", "nauseas", "vision",
    "bostezos_palidez",
]

DESENCADENANTES = [
    "estres", "sueno", "alimentos", "hormonal", "deshidratacion", "clima",
    "pantallas", "ayuno", "alcohol", "olores", "ejercicio", "ruido",
]

DURACIONES = ["menos_4h", "4_24h", "1_3d", "mas_3d"]

# ---------------------------------------------------------------------------
# Perfil (valores de la tabla perfil / cuestionario)
# ---------------------------------------------------------------------------
# Ataques al mes según la frecuencia que declara la persona.
# [SUPUESTO] punto medio de cada rango; 'casi_diario' por encima del umbral de
# migraña crónica de [ICHD3] (>=15 días/mes).
ATAQUES_MES = {
    "menos_1_mes": 0.6,
    "1_3_mes": 2.0,
    "1_2_semana": 6.0,
    "3_mas_semana": 13.0,
    "casi_diario": 20.0,
}

ORDEN_SUENO = {"menos_5": 0, "5_6": 1, "7_8": 2, "mas_8": 3}
ORDEN_ESTRES = {"bajo": 0, "moderado": 1, "alto": 2, "muy_alto": 3}
ORDEN_CAFEINA = {None: 0, "1": 1, "2_3": 2, "4_mas": 3}       # porciones al día
ORDEN_ANTIGUEDAD = {"menos_1_anio": 0, "1_3_anios": 1, "4_10_anios": 2, "mas_10_anios": 3}

# ---------------------------------------------------------------------------
# Síntomas premonitorios
# ---------------------------------------------------------------------------
# Probabilidad de que un ataque tenga aviso previo. [REVISION] 77-92% con buena
# pregunta; en la práctica la persona no siempre lo anota (ver CUMPLIMIENTO).
PROB_AVISO_POR_ATAQUE = (0.60, 0.90)          # rango entre personas

# Horas antes del dolor: [ICHD3] 2-48 h; [REVISION] media 6-10 h.
HORAS_AVISO = {"min": 2.0, "max": 48.0, "mediana": 8.0}

# Síntomas "tempranos" (aparecen 25-36 h antes) y "tardíos" (<=12 h). [REVISION]
SINTOMAS_TEMPRANOS = {"hiperactividad", "animo_exaltado", "avidez_alimentos", "hambre_atipica"}
SINTOMAS_TARDIOS = {"cervical", "cansancio", "astenia", "luz", "ruido", "nauseas", "vision"}

# Probabilidad de cada síntoma DENTRO de un aviso real.
# [ESTUDIO GIFFIN]: cansancio .72, concentración .51, cuello .50. El resto
# [SUPUESTO] según lo "frecuente" que lo describe [REVISION].
P_SINTOMA_EN_AVISO = {
    "cansancio": 0.72, "concentracion": 0.51, "cervical": 0.50,
    "bostezos_repetidos": 0.35, "bostezos_palidez": 0.20, "luz": 0.36,
    "ruido": 0.25, "astenia": 0.30, "hipoactividad": 0.22, "depresion": 0.15,
    "animo_exaltado": 0.12, "hiperactividad": 0.10, "avidez_alimentos": 0.18,
    "hambre_atipica": 0.15, "nauseas": 0.20, "vision": 0.12,
}

# Probabilidad de cada síntoma en un registro que NO termina en migraña (falsa
# alarma: cansancio normal, cuello por postura...). Así se reproduce [GIFFIN]:
# bostezos muy predictivos; cansancio, cuello y náuseas poco predictivos.
# [SUPUESTO] calibrado para que ~70% de los avisos acierten (cerca del 72% de GIFFIN).
P_SINTOMA_EN_FALSA_ALARMA = {
    "cansancio": 0.60, "concentracion": 0.30, "cervical": 0.45,
    "bostezos_repetidos": 0.06, "bostezos_palidez": 0.05, "luz": 0.12,
    "ruido": 0.10, "astenia": 0.20, "hipoactividad": 0.15, "depresion": 0.10,
    "animo_exaltado": 0.05, "hiperactividad": 0.05, "avidez_alimentos": 0.07,
    "hambre_atipica": 0.07, "nauseas": 0.15, "vision": 0.05,
}

# Falsas alarmas por mes (registros sin migraña después). [SUPUESTO] ajustado para
# que la precisión de los avisos ronde el 70% de [GIFFIN].
FALSAS_ALARMAS_MES = (0.3, 2.5)

# Intensidad anotada (1-10) del registro de síntomas. [SUPUESTO]
INTENSIDAD_AVISO = (2, 9)
INTENSIDAD_FALSA = (1, 7)

# ---------------------------------------------------------------------------
# Desencadenantes y modificadores del riesgo
# ---------------------------------------------------------------------------
# % de pacientes que reconocen cada desencadenante. [KELMAN]
# 'pantallas' = luces (38%), 'ayuno' = no comer (57%). Deshidratación y ruido no
# tienen cifra en KELMAN: [SUPUESTO] 30%.
PREVALENCIA_DESENCADENANTE = {
    "estres": 0.80, "hormonal": 0.65, "ayuno": 0.57, "clima": 0.53,
    "sueno": 0.50, "olores": 0.44, "pantallas": 0.38, "alcohol": 0.38,
    "alimentos": 0.27, "ejercicio": 0.22, "deshidratacion": 0.30, "ruido": 0.30,
}

# Multiplicadores del riesgo diario.
RIESGO_MENSTRUAL = {"dias_-2_-1": 1.71, "dias_1_5": 2.1}   # [MACGREGOR]
RIESGO_BAJADA_ESTRES_6H = 5.0                              # [LIPTON] (estudio chico)
RIESGO_CAFEINA_ALTA = 1.4                                  # [MOSTOFSKY] efecto sin cifra exacta -> [SUPUESTO] 1.4
# Perfil (sueño corto, estrés alto): [SUPUESTO] efectos moderados, 1.0 a 1.5
RIESGO_SUENO = {"menos_5": 1.45, "5_6": 1.2, "7_8": 1.0, "mas_8": 1.1}
RIESGO_ESTRES = {"bajo": 0.85, "moderado": 1.0, "alto": 1.25, "muy_alto": 1.5}

# Personas que no anotan todo. [SUPUESTO]
CUMPLIMIENTO_REGISTRO = (0.55, 0.95)   # prob. de anotar un aviso que sí sintió
CUMPLIMIENTO_EPISODIO = (0.80, 1.00)   # prob. de anotar un episodio

# Ventana que predice el modelo: ¿empieza una migraña en las próximas 48 h?
# [ICHD3] la fase premonitoria llega hasta 48 h antes.
HORIZONTE_HORAS = 48

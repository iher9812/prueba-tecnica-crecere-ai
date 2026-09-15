# Humanos vs. IA — Análisis de gestiones de cobranza

**Prueba Técnica — Data Scientist / Data Analyst Junior · Creceré AI**

Comparación cuantitativa del desempeño de agentes humanos vs. un agente de IA en
gestiones de cobranza, a partir de 100 llamadas reales (50 humanas + 50 IA).

📄 **Reporte ejecutivo:** [`report/reporte_final.html`](report/reporte_final.html)

---

## 1. Preguntas, hipótesis y decisiones analíticas

### ¿Qué quiero entender?

1. ¿Los agentes de IA logran resultados de cobranza (contactabilidad, compromisos
   de pago) comparables o mejores que los agentes humanos?
2. ¿Qué patrones de conducta — repetición de información, contradicciones, manejo
   de la confusión del cliente — diferencian a la IA de los humanos?
3. ¿Qué tan clara y eficiente es la comunicación de cada grupo?

### ¿Qué variables construí?

| Tipo | Variables | Cómo se obtienen |
|---|---|---|
| **Mecánicas** | duración, nº de turnos, tiempo de habla agente/cliente, palabras por minuto | directo de audio + transcripción |
| **De juicio** | contactabilidad, efectividad, negociación, compromiso de pago, objeciones, manejo de objeciones, resultado final, claridad | LLM lee la transcripción y responde un JSON estructurado |
| **Específicas del reto** | dato/pregunta repetida (conteo), contradicción o información incorrecta (sí/no), confusión del cliente y su resolución (bien/mal/no aplica) | mismo LLM — requieren comprensión semántica, no conteo de palabras |

Las variables de juicio no se pueden extraer con reglas simples (regex, conteo de
palabras) porque dependen de *entender* el diálogo — por eso se usa un LLM como
"lector" estructurado sobre cada transcripción (ver [Metodología](#3-metodología)).

### ¿Qué hipótesis contrasté?

- **H1:** la IA es al menos igual de efectiva que los humanos en obtener
  compromisos de pago.
- **H2:** la IA comete menos contradicciones / da menos información incorrecta
  que los humanos, por seguir un guion más consistente.
- **H3:** la IA repite más preguntas o datos que los humanos, por tener memoria
  conversacional más limitada.
- **H4:** los humanos manejan mejor la confusión del cliente que la IA, por
  mayor capacidad de adaptarse en tiempo real.

### ¿Cómo las probé?

Estadística descriptiva por grupo (humano vs. IA) + tests de hipótesis:
- **Mann-Whitney U** para variables continuas (duración, claridad, objeciones...).
- **Fisher exacto** para variables binarias (compromiso de pago, contradicción...).
- **Chi-cuadrado** para variables categóricas (manejo de objeciones, resolución
  de confusión).

Significancia al 5% (p < 0.05). Ver [`src/analysis.py`](src/analysis.py) y el
detalle completo en [`data/features/resultados_tests.csv`](data/features/resultados_tests.csv).

---

## 2. Hallazgos clave

Ver el reporte ejecutivo: [`report/reporte_final.html`](report/reporte_final.html).

**Contraste contra las hipótesis (sección 1):**

| Hipótesis | Resultado |
|---|---|
| H1: IA al menos tan efectiva como humanos en compromiso de pago | ❌ Rechazada — humanos logran el doble (42% vs. 20%, p=0.03) |
| H2: IA se contradice menos que los humanos | ❌ Rechazada — la IA se contradice muchísimo más (52% vs. 14%, p&lt;0.001) |
| H3: IA repite más preguntas/datos que los humanos | ➖ No confirmada — sin diferencia significativa (p=0.42 y p=0.77) |
| H4: humanos manejan mejor la confusión del cliente | ✅ Confirmada — 62% vs. 29% de confusiones bien resueltas (p=0.03) |

Dos de las cuatro hipótesis iniciales resultaron rechazadas por los datos — en particular, se esperaba
que la IA fuera más consistente que los humanos, y ocurrió lo contrario.

---

## 3. Metodología

```
Audios (.wav)
    │
    ▼
1. Transcripción + diarización  →  src/transcribe.py
   (faster-whisper + pyannote.audio, local, GPU)
    │
    ▼
2. Extracción de variables       →  src/extract_features.py
   (mecánicas + juicio vía LLM — API de Kimi)
    │
    ▼
3. Análisis estadístico          →  src/analysis.py
   (descriptivos + tests de hipótesis Humanos vs. IA)
    │
    ▼
4. Reporte HTML ejecutivo        →  src/build_report.py
```

**Por qué este enfoque:**
- **Transcripción local (faster-whisper) y diarización local (pyannote.audio):**
  100% privado y gratuito, corre sobre la GPU del equipo, sin exponer los audios
  a terceros.
- **Juicio semántico vía LLM (API de Kimi):** las tres preguntas específicas del
  reto (repetición, contradicción, confusión) exigen comprender el contenido de
  la conversación, no solo contarlo — un LLM actúa como lector estructurado,
  con un prompt fijo y schema JSON estricto para que el proceso sea reproducible.

## 4. Reproducir el pipeline

```bash
# 1. Entorno
python -m venv .venv
.venv/Scripts/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 2. Credenciales (copiar y completar, nunca commitear)
cp .env.example .env
# abrir .env y pegar KIMI_API_KEY y HF_TOKEN

# 3. Colocar los audios en data/raw/humanos/ y data/raw/ia/

# 4. Correr el pipeline completo
python src/transcribe.py
python src/extract_features.py
python src/analysis.py
python src/build_report.py
```

## 5. Estructura del repositorio

```
data/
  raw/{humanos,ia}/      # audios (no versionados — .gitignore)
  transcripts/            # transcripción + diarización por audio (JSON)
  features/                # dataset.csv + resultados_tests.csv
src/
  transcribe.py            # Fase 1
  extract_features.py      # Fase 2
  analysis.py               # Fase 3
  build_report.py           # Fase 5
report/
  reporte_final.html        # Entregable 01
```

## 6. Limitaciones

- Muestra de 50 + 50 llamadas: suficiente para tests no paramétricos, pero
  los hallazgos deben leerse como indicativos, no como verdad poblacional.
- La asignación de turnos "agente"/"cliente" usa una heurística sobre la
  diarización (quién domina la apertura de la llamada); se validó sobre una
  muestra manual (ver `src/transcribe.py`).
- Las variables de juicio dependen de la calidad del LLM usado como lector;
  se validaron manualmente sobre una muestra de transcripciones.

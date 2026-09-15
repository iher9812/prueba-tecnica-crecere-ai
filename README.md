# Humanos vs. IA — Análisis de gestiones de cobranza

**Prueba Técnica — Data Scientist / Data Analyst Junior · Creceré AI**

Comparación cuantitativa del desempeño de agentes humanos vs. un agente de IA en
gestiones de cobranza, a partir de 100 llamadas reales (50 humanas + 50 IA).

📄 **Reporte ejecutivo (entregable 01):** [`report/reporte_final.html`](report/reporte_final.html)

## Qué se hizo, paso a paso

1. **Transcribir** los 100 audios en local (GPU) y separar quién habla, agente o cliente — [`src/transcribe.py`](src/transcribe.py)
2. **Anonimizar** por reglas, sin IA ni red: 198 identificadores sustituidos por seudónimos — [`src/anonimizar.py`](src/anonimizar.py)
3. **Poner en cuarentena** las transcripciones crudas: solo la versión anonimizada sigue adelante, y el código lo impone
4. **Construir 21 variables** por llamada: 8 mecánicas del audio + 13 de juicio con un LLM — [`src/extract_features.py`](src/extract_features.py)
5. **Comparar** humanos vs. IA con Mann-Whitney U y Fisher exacto, corrigiendo por 17 comparaciones — [`src/analysis.py`](src/analysis.py)
6. **Reportar** en dos páginas, con la identidad visual del brief — [`src/build_report.py`](src/build_report.py)

**En una frase:** la IA se contradice más del doble que los humanos (44% vs. 18%) y cierra la
mitad de compromisos de pago (22% vs. 44%); la diferencia no está en cómo habla, sino en la
consistencia de lo que dice y en cómo repara la confusión del cliente.

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

Significancia al 5% (p < 0.05), con corrección por comparaciones múltiples
(Benjamini-Hochberg). Ver [`src/analysis.py`](src/analysis.py) y el detalle
completo en [`data/features/resultados_tests.csv`](data/features/resultados_tests.csv).

---

## 2. Hallazgos clave

Ver el reporte ejecutivo: [`report/reporte_final.html`](report/reporte_final.html).

**Contraste contra las hipótesis (sección 1):**

| Hipótesis | Resultado |
|---|---|
| H1: IA al menos tan efectiva como humanos en compromiso de pago | ❌ Rechazada — humanos logran el doble (44% vs. 22%, p=0.03) |
| H2: IA se contradice menos que los humanos | ❌ Rechazada — la IA se contradice mucho más (44% vs. 18%, p=0.009) |
| H3: IA repite más preguntas/datos que los humanos | ➖ No confirmada — sin diferencia significativa (p=0.09 y p=0.60) |
| H4: humanos manejan mejor la confusión del cliente | ✅ Confirmada — 67% vs. 34% de confusiones bien resueltas (p=0.02) |

Dos de las cuatro hipótesis iniciales resultaron rechazadas por los datos — en particular, se esperaba
que la IA fuera más consistente que los humanos, y ocurrió lo contrario.

**Advertencia estadística.** Se contrastaron 17 variables sobre la misma muestra. Al corregir por
comparaciones múltiples (Benjamini-Hochberg), **ningún hallazgo alcanza el umbral individualmente**
(mejor p ajustada = 0.076). Con n=50 por grupo el estudio está subpotenciado para establecer cada
efecto por separado. La lectura defendible es: **cuatro señales convergentes y coherentes entre sí**,
suficientes para orientar decisión y seguimiento, insuficientes para declararse concluyentes.

<details>
<summary><b>Estabilidad del juez LLM</b> — 90% test-retest, 100% en las variables que sostienen los hallazgos</summary>

El modelo se ejecuta con `temperature=1` (único valor que admite) y sin semilla, así que no es
determinista. Se midió estabilidad **test-retest** re-evaluando las mismas llamadas dos veces:
**90% de coincidencia global, 100% en las variables binarias** que sostienen los hallazgos. La
variación se concentró en escalas ordinales finas (`claridad`, `veces_repetido`, `manejo_objeciones`)
— justamente las que salieron no significativas. Ninguna conclusión descansa sobre lo inestable.

</details>

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
2. Anonimización                 →  src/anonimizar.py
   (reglas deterministas, local, sin IA)
    │
    ▼
3. Extracción de variables       →  src/extract_features.py
   (mecánicas + juicio vía LLM — API de Kimi)
    │
    ▼
4. Análisis estadístico          →  src/analysis.py
   (descriptivos + tests + corrección FDR)
    │
    ▼
5. Reporte HTML ejecutivo        →  src/build_report.py
```

**Por qué este enfoque:**
- **Transcripción local (faster-whisper) y diarización local (pyannote.audio):**
  100% privado y gratuito, corre sobre la GPU del equipo, sin exponer los audios
  a terceros.
- **Juicio semántico vía LLM (API de Kimi):** las tres preguntas específicas del
  reto (repetición, contradicción, confusión) exigen comprender el contenido de
  la conversación, no solo contarlo — un LLM actúa como lector estructurado,
  con un prompt fijo y schema JSON estricto para que el proceso sea reproducible.

---

## 4. Privacidad y arquitectura de datos

Los audios venían "censurados", pero el censurado era incompleto: sobrevivían ~168 menciones de
nombres de clientes. Todo lo que identifica a una persona se procesó en local y se anonimizó por
reglas antes de que cualquier texto saliera del equipo. El código lo impone: la única etapa que
llama a una API externa **aborta** si se le apunta a datos sin anonimizar.

<details>
<summary><b>Ver la cadena de tratamiento completa</b> — qué sale del equipo, qué no, y cómo se verificó</summary>

**Cadena de tratamiento de los datos:**

| # | Paso | Dónde corre | Resultado |
|---|---|---|---|
| 1 | Audio → transcripción y diarización | Local, GPU. Sin nube | `data/transcripts_crudas/` — conserva identificadores |
| 2 | Transcripción cruda → anonimización | Local, reglas deterministas. Sin IA, sin red | `data/transcripts_anonimizado/` |
| 3 | **Las crudas se separan y se excluyen del análisis** | — | Cuarentena: solo alimentan al paso 2 |
| 4 | **El análisis consume solo las anonimizadas** | Kimi (API) + local | dataset, tests, reporte |

Los pasos 3 y 4 no dependen de la disciplina de quien ejecuta.
[`extract_features.py`](src/extract_features.py) —la única etapa que envía texto fuera del
equipo— **aborta** si se le apunta a una carpeta que no sea la anonimizada. Cada carpeta de
datos lleva además su propio `LEEME.md`, versionado, explicando qué contiene y por qué no
se publica.

**Qué se queda local y qué no:**

| Activo | Procesamiento | ¿Sale del equipo? |
|---|---|---|
| Audio (6,4 h, 100 llamadas) | faster-whisper + pyannote, GPU local | No |
| Transcripciones con identificadores | Anonimizador por reglas, local | No |
| Transcripciones anonimizadas | API de Kimi (juicio semántico) | Sí — ya sin identificadores directos |
| Dataset agregado y reporte | Local | Publicados (verificados sin PII) |

**Retención.** Las transcripciones crudas se conservan localmente en lugar de destruirse:
tienen valor analítico y permiten reauditar la anonimización sin repetir ~2 h de GPU. El
control que corresponde a ese dato en un entorno real no es de código sino de despliegue
—cifrado de disco, control de acceso y un plazo de retención definido— y queda fuera del
alcance de este ejercicio, pero declarado.

**Por qué reglas deterministas y no NER.** El texto es español telefónico a 8 kHz, sin puntuación
fiable y con capitalización inconsistente — condiciones donde un modelo NER rinde mal. En cambio,
los nombres en gestión de cobranza aparecen con estructura altamente predecible
(`señor/señora/don/doña + X`, `mi nombre es X`, `le habla X`), que un patrón contextual captura con
alta precisión. Además, una regla es auditable línea por línea; un modelo no.

[`src/anonimizar.py`](src/anonimizar.py) aplica tres capas que se cubren mutuamente: patrón de
tratamiento, patrón de autopresentación y diccionario de nombres
([`data/diccionarios/nombres_es.txt`](data/diccionarios/nombres_es.txt), redactado desde conocimiento
general del español, **no** derivado de los datos). Los identificadores se sustituyen por seudónimos
consistentes dentro de cada llamada (`[NOMBRE_1]`), de modo que el diálogo sigue siendo coherente
para el análisis sin identificar a nadie. El mapa no se persiste: la seudonimización es irreversible.

**Resultado medido:** 198 identificadores reemplazados en 60 de 100 llamadas. **Cero menciones sin
seudonimizar tras un tratamiento** — el patrón de mayor riesgo quedó exhaustivo. Los residuos
restantes, revisados manualmente, son topónimos y nombres de entidades, no de personas.

**Diseño anti-exposición.** El anonimizador nunca imprime un valor detectado: a consola solo salen
conteos. Los valores en contexto se escriben en reportes HTML locales (excluidos de git) para
revisión humana. Esto permite auditar la calidad de la redacción sin exponer los datos a ningún
servicio ni modelo — incluido el asistente usado para desarrollar el proyecto.

**Chequeo de robustez.** El análisis se ejecutó sobre ambas versiones de los datos
([`src/comparar_datasets.py`](src/comparar_datasets.py)): la coincidencia media por variable es
81%, y **las conclusiones no cambian** — la brecha de contradicción, el doble de compromisos de pago
y la mejor reparación de confusión se sostienen en dirección y magnitud. Los hallazgos nunca
dependieron de información identificable.

</details>

---

## 5. Reproducir el pipeline

<details>
<summary><b>Ver los comandos</b> — entorno, credenciales y las cinco etapas</summary>

Requiere Python **3.12** (no 3.14: el ecosistema de audio aún no tiene ruedas compatibles) y una
GPU NVIDIA para la transcripción. Las versiones de `torch` van fijadas a propósito; ver la nota
en [`requirements.txt`](requirements.txt).

```bash
# 1. Entorno
py -3.12 -m venv .venv312
.venv312/Scripts/activate
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 2. Credenciales (copiar y completar, nunca commitear)
cp .env.example .env
# abrir .env y pegar KIMI_API_KEY y HF_TOKEN

# 3. Colocar los audios en data/raw/humanos/ y data/raw/ia/

# 4. Correr el pipeline completo
python src/transcribe.py        # transcripción + diarización (local, GPU)
python src/anonimizar.py        # anonimización por reglas (local, sin IA)
python src/extract_features.py  # variables mecánicas + juicio vía LLM
python src/analysis.py          # descriptivos + tests + corrección FDR
python src/build_report.py      # reporte ejecutivo HTML
```

</details>

## 6. Estructura del repositorio

<details>
<summary><b>Ver el árbol</b> — qué se versiona y qué permanece local</summary>

```
data/
  raw/{humanos,ia}/           # audios — local
  transcripts_crudas/          # EN CUARENTENA, con identificadores — local
    LEEME.md                   #   (versionado: explica qué va aquí y por qué no está)
  transcripts_anonimizado/     # única entrada válida del análisis — local
    LEEME.md                   #   (versionado)
  anonimizacion/               # reportes de auditoría de la redacción — local
  diccionarios/nombres_es.txt  # diccionario de nombres — versionado
  features/                    # dataset.csv, resultados_tests.csv — versionados
src/
  transcribe.py                # 1. transcripción + diarización
  watchdog_transcribe.py       #    salvaguarda de tiempo por archivo
  anonimizar.py                # 2. anonimización por reglas
  extract_features.py          # 3. construcción de variables
  analysis.py                  # 4. estadística + corrección por comparaciones múltiples
  comparar_datasets.py         #    chequeo de robustez pre/post anonimización
  build_report.py              # 5. generación del reporte
report/
  reporte_final.html           # Entregable 01
  assets/logo_crecere.b64      # logo embebido en el reporte
```

Solo se versionan código, diccionario público, resultados agregados y los `LEEME.md` que
documentan cada carpeta de datos. Audio, transcripciones —crudas y anonimizadas— y
reportes de auditoría permanecen locales.

</details>

## 7. Limitaciones

<details>
<summary><b>Ver las seis</b> — muestra, juez LLM, diarización y anonimización</summary>

- **Muestra de 50 + 50 llamadas.** Suficiente para tests no paramétricos, pero
  subpotenciada: ningún hallazgo sobrevive la corrección por comparaciones
  múltiples de forma individual (ver sección 2).
- **El juez LLM no se validó contra etiquetas humanas.** Se midió su estabilidad
  test-retest (90%) y se inspeccionaron casos cualitativamente, pero no se
  construyó un *gold standard* anotado por una persona. Es el siguiente paso natural.
- **`efectividad` y `compromiso_pago` resultaron idénticas** en las 100 llamadas: el
  modelo colapsó ambos constructos. Se eliminó la redundante en vez de reportar el
  mismo hallazgo dos veces.
- **La asignación agente/cliente** usa una heurística sobre la diarización (quién
  domina la apertura); validada por muestreo manual, no exhaustivamente.
- **Cuatro llamadas quedaron con un solo hablante detectado.** Tres son llamadas
  reales sin respuesta del cliente (dato válido); una tiene audio de baja calidad
  que impidió separar voces.
- **La anonimización por reglas no garantiza recall perfecto.** Se mide y se reporta
  (sección 4) en lugar de afirmar anonimización total.

</details>

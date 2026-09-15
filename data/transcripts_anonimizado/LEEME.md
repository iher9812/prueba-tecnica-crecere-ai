# Transcripciones anonimizadas — única entrada válida del análisis

**Esta carpeta está vacía en el repositorio a propósito.** Su contenido existe solo en el
equipo donde se ejecutó el pipeline.

## Qué va aquí

La salida de [`src/anonimizar.py`](../../src/anonimizar.py): las mismas transcripciones de
`data/transcripts_crudas/`, con los identificadores directos sustituidos por seudónimos
consistentes dentro de cada llamada (`[NOMBRE_1]`, `[DOCUMENTO_1]`, `[REFERENCIA_1]`).

La consistencia por llamada es deliberada: el diálogo sigue siendo coherente para el
análisis —se puede ver que el agente trata al cliente por su nombre— sin que ese nombre sea
recuperable. El mapa de sustitución no se persiste, así que la seudonimización es
irreversible.

## Por qué sigue siendo sensible

**La anonimización por reglas no garantiza recall perfecto.** Se midió y se reporta en vez
de afirmar anonimización total: 198 identificadores reemplazados en 60 de 100 llamadas,
cero menciones sin seudonimizar tras un tratamiento (`señor/señora/don/doña`), y los
residuos revisados manualmente resultaron ser topónimos y nombres de entidades, no de
personas.

Aun así puede quedar algo: un nombre poco común, sin tratamiento previo, fuera del
diccionario y distorsionado por la transcripción automática. Por eso estos archivos
tampoco se versionan.

## Quién la consume

Todas las etapas posteriores del pipeline, empezando por
[`src/extract_features.py`](../../src/extract_features.py) — que es la única que envía texto
fuera del equipo, y que **aborta** si se le apunta a cualquier carpeta que no sea esta.

## Cómo regenerarlas

```bash
python src/anonimizar.py
```

Genera además dos reportes de auditoría locales en `data/anonimizacion/` —
tampoco versionados— para revisar la calidad de la redacción sin exponer los valores a
ningún servicio ni modelo.

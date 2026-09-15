# Transcripciones crudas — en cuarentena

**Esta carpeta está vacía en el repositorio a propósito.** Su contenido existe solo en el
equipo donde se ejecutó el pipeline y nunca se versiona.

## Qué va aquí

La salida de [`src/transcribe.py`](../../src/transcribe.py): un JSON por llamada con la
transcripción y la diarización, **conservando los identificadores tal como aparecen en el
audio original** — nombres de clientes, y en su caso documentos o teléfonos.

El audio de origen venía "censurado", pero la anonimización era incompleta: al auditarlo,
solo 21 de 100 transcripciones tenían marcas de censura y sobrevivían ~168 menciones de
nombres. Por eso estos archivos se tratan como datos personales sensibles: son
conversaciones de cobranza, donde la identidad va unida a información de mora.

## Por qué está separada

Es el punto 3 de la cadena de tratamiento:

1. Audio → transcripción, **local**
2. Transcripción cruda → anonimización, **local y por reglas**
3. **Las transcripciones crudas se separan y se excluyen del análisis** ← esta carpeta
4. El análisis consume **solo** `data/transcripts_anonimizado/`

Esta carpeta alimenta **únicamente** a [`src/anonimizar.py`](../../src/anonimizar.py).
Ningún otro script la lee.

La separación no depende de la disciplina de quien ejecuta:
[`src/extract_features.py`](../../src/extract_features.py) —la única etapa que envía texto a
una API externa— **aborta** si se le apunta a una carpeta que no sea la anonimizada.

## Retención

Los datos se conservan localmente porque tienen valor analítico: permiten reauditar la
anonimización o reprocesar sin repetir ~2 h de transcripción en GPU. En un entorno real
corresponde protegerlos con cifrado de disco y control de acceso, y fijarles un plazo de
retención acorde a la política de tratamiento de datos de la organización.

## Cómo regenerarlas

```bash
python src/transcribe.py
```

Requiere los audios en `data/raw/{humanos,ia}/` y un `HF_TOKEN` válido en `.env`.

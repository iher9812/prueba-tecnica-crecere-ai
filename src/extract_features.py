"""
Fase 2 — Extracción de variables.

Combina:
  (a) variables mecánicas, calculadas directo de cada JSON de transcripción
      (data/transcripts/*.json), y
  (b) variables de juicio, extraídas con un LLM (API de Kimi) que lee cada
      transcripción y responde un JSON estructurado.

Guarda el dataset final en data/features/dataset.csv.

Uso:
    python src/extract_features.py
"""
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = ROOT / "data" / "transcripts"
OUT_PATH = ROOT / "data" / "features" / "dataset.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

KIMI_API_KEY = os.getenv("KIMI_API_KEY")
KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
KIMI_MODEL = os.getenv("KIMI_MODEL", "kimi-k2-0711-preview")

SYSTEM_PROMPT = """\
Eres un analista de calidad de llamadas de un centro de gestión de cobranza.
Vas a leer la transcripción de UNA llamada entre un Agente y un Cliente.
Los turnos vienen marcados como [agente] o [cliente].

Responde ÚNICAMENTE con un JSON válido (sin texto adicional, sin markdown),
con exactamente estas claves:

{
  "contactabilidad": bool,        // ¿el cliente atendió y sostuvo la conversación?
  "efectividad": bool,            // ¿se logró el objetivo de la gestión (pago o compromiso)?
  "hubo_negociacion": bool,       // ¿el agente ofreció alternativas (plazos, descuentos, planes)?
  "compromiso_pago": bool,        // ¿el cliente se comprometió a pagar?
  "objeciones_count": int,        // cuántas objeciones distintas puso el cliente
  "manejo_objeciones": "bueno" | "regular" | "malo" | "no_aplica",
  "resultado_final": string,      // 3-6 palabras describiendo el desenlace
  "claridad": int,                // 1-5, qué tan claro/entendible fue el agente
  "dato_repetido": bool,          // ¿el agente repitió una pregunta o dato (aunque sea parafraseado)?
  "veces_repetido": int,          // cuántas veces se repitió (0 si no aplica)
  "contradiccion": bool,          // ¿el agente se contradijo o dio información incorrecta?
  "confusion_cliente": bool,      // ¿hubo un momento de confusión del cliente?
  "confusion_resuelta": "bien" | "mal" | "no_aplica"
}

Sé estricto: si no hay evidencia clara en el texto, usa el valor más conservador
(false / 0 / "no_aplica"). No inventes información que no esté en la transcripción.
"""


def call_kimi(texto_completo: str, max_retries: int = 3) -> dict:
    headers = {"Authorization": f"Bearer {KIMI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": KIMI_MODEL,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": texto_completo},
        ],
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{KIMI_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=60
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(content)
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def mechanical_features(transcript: dict) -> dict:
    turnos = transcript["turnos"]
    agente_turnos = [t for t in turnos if t["rol"] == "agente"]
    cliente_turnos = [t for t in turnos if t["rol"] == "cliente"]

    agente_tiempo = sum(t["fin"] - t["inicio"] for t in agente_turnos)
    cliente_tiempo = sum(t["fin"] - t["inicio"] for t in cliente_turnos)
    agente_palabras = sum(len(t["texto"].split()) for t in agente_turnos)

    duracion_min = transcript["duracion_seg"] / 60 if transcript["duracion_seg"] else 0

    return {
        "id": transcript["id"],
        "grupo": transcript["grupo"],
        "duracion_seg": transcript["duracion_seg"],
        "num_turnos": len(turnos),
        "num_turnos_agente": len(agente_turnos),
        "num_turnos_cliente": len(cliente_turnos),
        "tiempo_agente_seg": agente_tiempo,
        "tiempo_cliente_seg": cliente_tiempo,
        "proporcion_habla_agente": agente_tiempo / (agente_tiempo + cliente_tiempo)
        if (agente_tiempo + cliente_tiempo) > 0
        else None,
        "palabras_por_minuto_agente": agente_palabras / duracion_min if duracion_min > 0 else None,
    }


def main():
    if not KIMI_API_KEY:
        raise SystemExit(
            "Falta KIMI_API_KEY en .env. Abre el archivo .env y pega tu clave ahí "
            "(nunca la escribas en el chat ni la commitees)."
        )

    rows = []
    transcript_files = sorted(TRANSCRIPTS_DIR.glob("*.json"))
    for path in tqdm(transcript_files, desc="Extrayendo variables"):
        transcript = json.loads(path.read_text(encoding="utf-8"))
        row = mechanical_features(transcript)
        try:
            juicio = call_kimi(transcript["texto_completo"])
            row.update(juicio)
        except Exception as e:
            row["_error_kimi"] = str(e)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Dataset guardado en {OUT_PATH} ({len(df)} filas)")


if __name__ == "__main__":
    main()

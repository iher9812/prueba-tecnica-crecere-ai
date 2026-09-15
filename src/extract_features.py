"""
Fase 2 — Extracción de variables.

Combina:
  (a) variables mecánicas, calculadas directo de cada JSON de transcripción
      (data/transcripts/*.json), y
  (b) variables de juicio, extraídas con un LLM (API de Kimi) que lee cada
      transcripción y responde un JSON estructurado.

Kimi (kimi-k2.6) es un modelo de razonamiento: antes de responder genera
varios miles de tokens de "pensamiento" interno, así que cada llamada tarda
~1-2 minutos. Por eso las llamadas se hacen en paralelo (varios hilos) y el
progreso se guarda incrementalmente en un checkpoint — si el proceso se
interrumpe, al reiniciar retoma solo lo que falta.

Guarda el dataset final en data/features/dataset.csv.

Uso:
    python src/extract_features.py
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = ROOT / "data" / "transcripts"
CHECKPOINT_DIR = ROOT / "data" / "features" / "checkpoint"
OUT_PATH = ROOT / "data" / "features" / "dataset.csv"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

KIMI_API_KEY = os.getenv("KIMI_API_KEY")
KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
KIMI_MODEL = os.getenv("KIMI_MODEL", "kimi-k2.6")
MAX_WORKERS = int(os.getenv("KIMI_MAX_WORKERS", "6"))

SYSTEM_PROMPT = """\
Eres un analista de calidad de llamadas de un centro de gestión de cobranza.
Vas a leer la transcripción de UNA llamada entre un Agente y un Cliente.
Los turnos vienen marcados como [agente] o [cliente]. Algunas llamadas no
tienen turnos de cliente porque nadie respondió (buzón, sin respuesta) — en
ese caso usa el valor más conservador para las variables que dependan del
cliente.

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
        # kimi-k2.6/k3 solo admiten temperature=1 (rechazan otros valores);
        # la consistencia del formato se logra con el schema JSON estricto
        # del prompt, no bajando la temperatura.
        "temperature": 1,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": texto_completo},
        ],
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{KIMI_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=240
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(content)
        except (requests.RequestException, json.JSONDecodeError, KeyError):
            if attempt == max_retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


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


def process_one(path: Path) -> dict:
    transcript = json.loads(path.read_text(encoding="utf-8"))
    row = mechanical_features(transcript)
    try:
        juicio = call_kimi(transcript["texto_completo"])
        row.update(juicio)
    except Exception as e:
        row["_error_kimi"] = str(e)
    return row


def main():
    if not KIMI_API_KEY:
        raise SystemExit(
            "Falta KIMI_API_KEY en .env. Abre el archivo .env y pega tu clave ahí "
            "(nunca la escribas en el chat ni la commitees)."
        )

    transcript_files = sorted(TRANSCRIPTS_DIR.glob("*.json"))
    pendientes = []
    for path in transcript_files:
        checkpoint_path = CHECKPOINT_DIR / f"{path.stem}.json"
        if not checkpoint_path.exists():
            pendientes.append(path)

    print(f"{len(transcript_files) - len(pendientes)} ya en checkpoint, "
          f"{len(pendientes)} por procesar (con {MAX_WORKERS} hilos en paralelo).")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one, path): path for path in pendientes}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extrayendo variables (Kimi)"):
            path = futures[future]
            row = future.result()
            checkpoint_path = CHECKPOINT_DIR / f"{path.stem}.json"
            checkpoint_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = [
        json.loads((CHECKPOINT_DIR / f"{path.stem}.json").read_text(encoding="utf-8"))
        for path in transcript_files
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Dataset guardado en {OUT_PATH} ({len(df)} filas)")


if __name__ == "__main__":
    main()

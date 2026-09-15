"""
Fase 1 — Transcripción + diarización.

Recorre los audios en data/raw/{humanos,ia}/, transcribe cada uno con
faster-whisper y separa turnos agente/cliente con pyannote.audio.
Guarda un JSON por audio en data/transcripts/.

Uso:
    python src/transcribe.py
"""
import json
import os
from pathlib import Path

import torch

import torchaudio
import huggingface_hub
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from tqdm import tqdm

# --- Shims de compatibilidad: pyannote.audio==3.3.2 es de 2024 y usa APIs
# que las versiones recientes de sus dependencias ya eliminaron/renombraron.

# torchaudio >= 2.9 eliminó `list_audio_backends`; pyannote todavía lo llama
# para elegir el backend de audio. Forzamos "soundfile" (ya instalado).
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

# huggingface_hub >= 1.0 renombró `use_auth_token` a `token` en hf_hub_download.
_original_hf_hub_download = huggingface_hub.hf_hub_download


def _hf_hub_download_compat(*args, **kwargs):
    if "use_auth_token" in kwargs:
        kwargs["token"] = kwargs.pop("use_auth_token")
    return _original_hf_hub_download(*args, **kwargs)


huggingface_hub.hf_hub_download = _hf_hub_download_compat

from pyannote.audio import Pipeline

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
RAW_DIRS = {
    "humano": ROOT / "data" / "raw" / "humanos",
    "ia": ROOT / "data" / "raw" / "ia",
}
OUT_DIR = ROOT / "data" / "transcripts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL = "medium"  # balance velocidad/calidad para 6+ horas de audio
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"  # cabe cómodo en 8GB VRAM

HF_TOKEN = os.getenv("HF_TOKEN")


def load_models():
    whisper_model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN
    ).to(torch.device("cuda"))
    return whisper_model, diarization_pipeline


def transcribe_audio(whisper_model, audio_path: Path):
    segments, info = whisper_model.transcribe(
        str(audio_path), language="es", vad_filter=True
    )
    return [
        {"inicio": seg.start, "fin": seg.end, "texto": seg.text.strip()}
        for seg in segments
    ], info.duration


def diarize_audio(diarization_pipeline, audio_path: Path):
    # Todas las llamadas son de exactamente 2 personas (agente + cliente).
    # Sin esta pista, pyannote a veces fusiona a ambos hablantes en uno solo
    # cuando sus voces son parecidas o el audio es de baja calidad (telefonía).
    diarization = diarization_pipeline(str(audio_path), num_speakers=2)
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({"inicio": turn.start, "fin": turn.end, "hablante_id": speaker})
    return turns


def assign_speaker(segment, diarization_turns):
    """Asigna el hablante de pyannote cuyo turno más se solapa con el segmento."""
    best_overlap, best_speaker = 0.0, "desconocido"
    for turn in diarization_turns:
        overlap = min(segment["fin"], turn["fin"]) - max(segment["inicio"], turn["inicio"])
        if overlap > best_overlap:
            best_overlap, best_speaker = overlap, turn["hablante_id"]
    return best_speaker


def label_agente_cliente(segments, primeros_n_turnos=8):
    """
    Heurística: el hablante_id que más habla en los primeros turnos de la
    llamada (no por marca de tiempo absoluta, porque puede haber silencio/
    timbre al inicio) suele ser el agente, quien abre la gestión. Se valida
    manualmente sobre una muestra (ver README, sección Verificación).
    """
    apertura = segments[:primeros_n_turnos]
    conteo = {}
    for s in apertura:
        conteo[s["hablante_id"]] = conteo.get(s["hablante_id"], 0) + (s["fin"] - s["inicio"])
    if not conteo:
        return {}
    agente_id = max(conteo, key=conteo.get)
    ids = {s["hablante_id"] for s in segments}
    return {hid: ("agente" if hid == agente_id else "cliente") for hid in ids}


def process_file(whisper_model, diarization_pipeline, audio_path: Path, grupo: str):
    segments, duracion = transcribe_audio(whisper_model, audio_path)
    diarization_turns = diarize_audio(diarization_pipeline, audio_path)

    for seg in segments:
        seg["hablante_id"] = assign_speaker(seg, diarization_turns)

    roles = label_agente_cliente(segments)
    for seg in segments:
        seg["rol"] = roles.get(seg["hablante_id"], "desconocido")

    texto_completo = "\n".join(f"[{s['rol']}] {s['texto']}" for s in segments)

    return {
        "id": audio_path.stem,
        "grupo": grupo,
        "duracion_seg": duracion,
        "turnos": segments,
        "texto_completo": texto_completo,
    }


def main():
    whisper_model, diarization_pipeline = load_models()

    for grupo, folder in RAW_DIRS.items():
        audio_files = sorted(folder.glob("*.wav"))
        for audio_path in tqdm(audio_files, desc=f"Transcribiendo {grupo}"):
            out_path = OUT_DIR / f"{audio_path.stem}.json"
            if out_path.exists():
                continue  # checkpoint: no reprocesar si ya existe
            result = process_file(whisper_model, diarization_pipeline, audio_path, grupo)
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

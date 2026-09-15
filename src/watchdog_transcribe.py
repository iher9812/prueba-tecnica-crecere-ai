"""
Corre transcribe.py con un límite de tiempo por archivo.

El costo de la diarización de pyannote no es proporcional a la duración del
audio: depende del número de segmentos de habla, así que una llamada con
muchos turnos cortos puede tardar 15+ minutos mientras otra más larga tarda
uno. Sin una salvaguarda, un solo archivo patológico bloquea el lote entero.

Este watchdog vigila el avance sobre data/transcripts_crudas/. Si pasan más de
TIMEOUT_SEG sin que aparezca un archivo nuevo, mata el proceso, marca el
archivo atascado como fallido — para que el checkpoint de transcribe.py lo
salte — y relanza para continuar con el resto.

Uso:
    python src/watchdog_transcribe.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIRS = {"humano": ROOT / "data" / "raw" / "humanos", "ia": ROOT / "data" / "raw" / "ia"}
OUT_DIR = ROOT / "data" / "transcripts_crudas"
PYTHON = ROOT / ".venv312" / "Scripts" / "python.exe"
SCRIPT = ROOT / "src" / "transcribe.py"

TIMEOUT_SEG = 1500  # 25 min sin progreso => se considera atascado
# (algunos audios con muchos turnos cortos hacen que el clustering de
# pyannote sea legítimamente lento, hasta 15-18 min en casos observados;
# 25 min da margen sin dejar que un caso realmente patológico bloquee todo)
POLL_SEG = 10

archivos_saltados = []


def lista_ordenada():
    pares = []
    for grupo, folder in RAW_DIRS.items():
        for audio_path in sorted(folder.glob("*.wav")):
            pares.append((grupo, audio_path))
    return pares


def siguiente_pendiente(pares):
    for grupo, audio_path in pares:
        out_path = OUT_DIR / f"{audio_path.stem}.json"
        if not out_path.exists():
            return grupo, audio_path
    return None


def marcar_fallido(grupo, audio_path, motivo):
    out_path = OUT_DIR / f"{audio_path.stem}.json"
    out_path.write_text(
        json.dumps(
            {
                "id": audio_path.stem,
                "grupo": grupo,
                "duracion_seg": None,
                "turnos": [],
                "texto_completo": "",
                "error": motivo,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    archivos_saltados.append(audio_path.stem)
    print(f"[watchdog] Marcado como fallido ({motivo}): {audio_path.name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pares = lista_ordenada()
    total = len(pares)

    while True:
        pendiente = siguiente_pendiente(pares)
        if pendiente is None:
            print(f"[watchdog] Completo. {total - len(archivos_saltados)} ok, "
                  f"{len(archivos_saltados)} saltados: {archivos_saltados}")
            break

        print(f"[watchdog] Lanzando transcribe.py (siguiente pendiente: {pendiente[1].name})")
        proc = subprocess.Popen(
            [str(PYTHON), str(SCRIPT)],
            cwd=str(ROOT / "src"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        last_count = len(list(OUT_DIR.glob("*.json")))
        last_progress_time = time.time()

        while True:
            time.sleep(POLL_SEG)
            current_count = len(list(OUT_DIR.glob("*.json")))
            if current_count > last_count:
                last_count = current_count
                last_progress_time = time.time()

            if proc.poll() is not None:
                break  # el proceso terminó (bien o con error)

            if time.time() - last_progress_time > TIMEOUT_SEG:
                grupo, audio_path = siguiente_pendiente(pares)
                print(f"[watchdog] Sin progreso por {TIMEOUT_SEG}s. Matando proceso "
                      f"(atascado en: {audio_path.name})")
                proc.kill()
                proc.wait()
                marcar_fallido(grupo, audio_path, "timeout_diarizacion")
                break


if __name__ == "__main__":
    main()

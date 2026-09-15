"""
Chequeo de robustez: ¿cambian los resultados al anonimizar?

Compara el dataset generado sobre transcripciones crudas contra el generado
sobre transcripciones anonimizadas. Si los hallazgos se sostienen, es evidencia
de que nunca dependieron de información identificable.

Compara únicamente variables derivadas (etiquetas y métricas), nunca texto:
su salida es segura de compartir.

Uso:
    python src/comparar_datasets.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANTES = ROOT / "data" / "features" / "dataset_pre_anonimizacion.csv"
DESPUES = ROOT / "data" / "features" / "dataset.csv"
SALIDA = ROOT / "data" / "features" / "comparacion_anonimizacion.csv"

JUICIO = [
    "contactabilidad", "efectividad", "hubo_negociacion", "compromiso_pago",
    "objeciones_count", "manejo_objeciones", "claridad", "dato_repetido",
    "veces_repetido", "contradiccion", "confusion_cliente", "confusion_resuelta",
]
TITULARES = ["contradiccion", "compromiso_pago", "contactabilidad", "hubo_negociacion"]


def main():
    antes = pd.read_csv(ANTES).set_index("id").sort_index()
    despues = pd.read_csv(DESPUES).set_index("id").sort_index()
    comunes = antes.index.intersection(despues.index)
    antes, despues = antes.loc[comunes], despues.loc[comunes]

    filas = []
    for col in JUICIO:
        if col in antes.columns and col in despues.columns:
            coincide = (antes[col] == despues[col]).mean()
            filas.append({"variable": col, "coincidencia": round(coincide, 3)})
    tabla = pd.DataFrame(filas).sort_values("coincidencia", ascending=False)

    print(f"Llamadas comparadas: {len(comunes)}")
    print(f"\n--- Coincidencia por variable (crudo vs. anonimizado) ---")
    print(tabla.to_string(index=False))
    print(f"\nCoincidencia media: {tabla['coincidencia'].mean():.1%}")

    print(f"\n--- ¿Se sostienen los hallazgos? (tasas por grupo) ---")
    resumen = []
    for col in TITULARES:
        if col not in antes.columns:
            continue
        for grupo in ["humano", "ia"]:
            a = antes.loc[antes.grupo == grupo, col].astype(bool).mean()
            d = despues.loc[despues.grupo == grupo, col].astype(bool).mean()
            resumen.append({
                "variable": col, "grupo": grupo,
                "antes": f"{a:.0%}", "despues": f"{d:.0%}",
                "delta_pp": round((d - a) * 100, 1),
            })
    resumen = pd.DataFrame(resumen)
    print(resumen.to_string(index=False))

    tabla.to_csv(SALIDA, index=False, encoding="utf-8")
    print(f"\nGuardado en {SALIDA}")


if __name__ == "__main__":
    main()

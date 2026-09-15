"""
Fase 3 — Estadística descriptiva y comparación Humanos vs IA.

Lee data/features/dataset.csv y produce data/features/resultados_tests.csv
con: variable, valor humano, valor IA, diferencia, p-valor, tipo de test.

Uso:
    python src/analysis.py
"""
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "features" / "dataset.csv"
OUT_PATH = ROOT / "data" / "features" / "resultados_tests.csv"

BOOL_COLS = [
    "contactabilidad", "efectividad", "hubo_negociacion", "compromiso_pago",
    "dato_repetido", "contradiccion", "confusion_cliente",
]
CONTINUOUS_COLS = [
    "duracion_seg", "num_turnos", "proporcion_habla_agente",
    "palabras_por_minuto_agente", "objeciones_count", "veces_repetido", "claridad",
]
CATEGORICAL_COLS = ["manejo_objeciones", "confusion_resuelta"]


def compare_boolean(df, col):
    humano = df.loc[df.grupo == "humano", col].dropna().astype(bool)
    ia = df.loc[df.grupo == "ia", col].dropna().astype(bool)
    prop_h, prop_ia = humano.mean(), ia.mean()
    table = [[humano.sum(), len(humano) - humano.sum()], [ia.sum(), len(ia) - ia.sum()]]
    _, p = stats.fisher_exact(table)
    return {
        "variable": col, "tipo_test": "fisher_exact",
        "valor_humano": prop_h, "valor_ia": prop_ia,
        "diferencia": prop_ia - prop_h, "p_valor": p,
        "significativo_0.05": p < 0.05,
    }


def compare_continuous(df, col):
    humano = df.loc[df.grupo == "humano", col].dropna()
    ia = df.loc[df.grupo == "ia", col].dropna()
    _, p = stats.mannwhitneyu(humano, ia, alternative="two-sided")
    return {
        "variable": col, "tipo_test": "mann_whitney_u",
        "valor_humano": humano.mean(), "valor_ia": ia.mean(),
        "diferencia": ia.mean() - humano.mean(), "p_valor": p,
        "significativo_0.05": p < 0.05,
    }


def compare_categorical(df, col):
    tabla = pd.crosstab(df["grupo"], df[col])
    _, p, _, _ = stats.chi2_contingency(tabla)
    moda_h = df.loc[df.grupo == "humano", col].mode().iloc[0]
    moda_ia = df.loc[df.grupo == "ia", col].mode().iloc[0]
    return {
        "variable": col, "tipo_test": "chi2_contingency",
        "valor_humano": moda_h, "valor_ia": moda_ia,
        "diferencia": None, "p_valor": p,
        "significativo_0.05": p < 0.05,
    }


def main():
    df = pd.read_csv(DATASET_PATH)
    if "_error_kimi" in df.columns:
        fallidos = df["_error_kimi"].notna().sum()
        if fallidos:
            print(f"Aviso: {fallidos} filas sin datos de Kimi (fallaron en extracción).")

    resultados = []
    for col in BOOL_COLS:
        if col in df.columns:
            resultados.append(compare_boolean(df, col))
    for col in CONTINUOUS_COLS:
        if col in df.columns:
            resultados.append(compare_continuous(df, col))
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            resultados.append(compare_categorical(df, col))

    resultados_df = pd.DataFrame(resultados).sort_values("p_valor")
    resultados_df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Resultados guardados en {OUT_PATH}")
    print(resultados_df.to_string(index=False))


if __name__ == "__main__":
    main()

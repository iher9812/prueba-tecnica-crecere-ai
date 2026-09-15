"""
Fase 5 — Genera el reporte HTML ejecutivo (entregable 01).

Lee data/features/dataset.csv y data/features/resultados_tests.csv,
genera gráficos y arma un único HTML autocontenido (máx. 2 páginas)
en report/reporte_final.html, reutilizando la paleta de marca de
Creceré AI (rosa/morado) del brief original.

Uso:
    python src/build_report.py
"""
import base64
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "features" / "dataset.csv"
RESULTS_PATH = ROOT / "data" / "features" / "resultados_tests.csv"
OUT_PATH = ROOT / "report" / "reporte_final.html"

PINK = "#F26ECF"
PURPLE = "#7047EB"
DEEP = "#4E2AAE"
INK = "#202020"
MUTED = "#757275"

VARIABLE_LABELS = {
    "efectividad": "Efectividad (objetivo logrado)",
    "compromiso_pago": "Compromiso de pago obtenido",
    "contactabilidad": "Contactabilidad",
    "hubo_negociacion": "Hubo negociación",
    "dato_repetido": "Repitió una pregunta o dato",
    "contradiccion": "Se contradijo / dio info incorrecta",
    "confusion_cliente": "Hubo confusión del cliente",
    "claridad": "Claridad (1-5)",
    "duracion_seg": "Duración (seg)",
    "objeciones_count": "Objeciones por llamada",
    "veces_repetido": "Veces que repitió un dato",
}


def fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def bar_comparison(resultados: pd.DataFrame, cols: list[str], title: str, as_pct: bool):
    sub = resultados[resultados["variable"].isin(cols)].set_index("variable").loc[cols]
    labels = [VARIABLE_LABELS.get(c, c) for c in cols]
    humano_vals = sub["valor_humano"].astype(float)
    ia_vals = sub["valor_ia"].astype(float)
    if as_pct:
        humano_vals, ia_vals = humano_vals * 100, ia_vals * 100

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    y = range(len(labels))
    height = 0.32
    ax.barh([i + height / 2 for i in y], humano_vals, height=height, color=MUTED, label="Humanos")
    ax.barh([i - height / 2 for i in y], ia_vals, height=height, color=PURPLE, label="IA")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("%" if as_pct else "valor")
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig_to_base64(fig)


def format_hallazgo(row) -> str:
    var = VARIABLE_LABELS.get(row["variable"], row["variable"])
    if row["tipo_test"] in ("fisher_exact",):
        h, ia = row["valor_humano"] * 100, row["valor_ia"] * 100
        return f"<b>{var}:</b> IA {ia:.0f}% vs. Humanos {h:.0f}% (dif. {ia - h:+.0f} pp, p={row['p_valor']:.3f})"
    if row["tipo_test"] == "mann_whitney_u":
        return f"<b>{var}:</b> IA {row['valor_ia']:.1f} vs. Humanos {row['valor_humano']:.1f} (p={row['p_valor']:.3f})"
    return f"<b>{var}:</b> Humanos → {row['valor_humano']} · IA → {row['valor_ia']} (p={row['p_valor']:.3f})"


def build_hallazgos(resultados: pd.DataFrame, n=6) -> list[str]:
    significativos = resultados[resultados["significativo_0.05"]].copy()
    significativos = significativos.sort_values("p_valor").head(n)
    return [format_hallazgo(r) for _, r in significativos.iterrows()]


def kpi_cards(dataset: pd.DataFrame, resultados: pd.DataFrame) -> str:
    n_humano = (dataset.grupo == "humano").sum()
    n_ia = (dataset.grupo == "ia").sum()

    def get(var):
        row = resultados[resultados.variable == var]
        return row.iloc[0] if len(row) else None

    cards = []
    efect = get("efectividad")
    if efect is not None:
        cards.append(("Efectividad IA", f"{efect.valor_ia*100:.0f}%", f"vs {efect.valor_humano*100:.0f}% humanos"))
    pago = get("compromiso_pago")
    if pago is not None:
        cards.append(("Compromiso de pago IA", f"{pago.valor_ia*100:.0f}%", f"vs {pago.valor_humano*100:.0f}% humanos"))
    contra = get("contradiccion")
    if contra is not None:
        cards.append(("Contradicciones IA", f"{contra.valor_ia*100:.0f}%", f"vs {contra.valor_humano*100:.0f}% humanos"))
    cards.append(("Llamadas analizadas", f"{n_humano + n_ia}", f"{n_humano} humanas + {n_ia} IA"))

    html = ""
    for titulo, valor, sub in cards:
        html += f"""
        <div class="kpi">
          <div class="kpi-label">{titulo}</div>
          <div class="kpi-value">{valor}</div>
          <div class="kpi-sub">{sub}</div>
        </div>"""
    return html


def main():
    dataset = pd.read_csv(DATASET_PATH)
    resultados = pd.read_csv(RESULTS_PATH)

    img_desempeno = bar_comparison(
        resultados, ["efectividad", "compromiso_pago", "contactabilidad", "hubo_negociacion"],
        "Desempeño Humanos vs. IA", as_pct=True,
    )
    img_conducta = bar_comparison(
        resultados, ["dato_repetido", "contradiccion", "confusion_cliente"],
        "Conducta problemática Humanos vs. IA", as_pct=True,
    )

    hallazgos = build_hallazgos(resultados)
    hallazgos_html = "".join(f"<li>{h}</li>" for h in hallazgos)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8" />
<title>Humanos vs. IA — Reporte Ejecutivo | Creceré AI</title>
<style>
  :root{{ --pink:{PINK}; --purple:{PURPLE}; --deep:{DEEP}; --ink:{INK}; --muted:{MUTED}; }}
  *{{box-sizing:border-box}}
  body{{margin:0;font-family:Inter,ui-sans-serif,Arial,sans-serif;color:var(--ink);background:#f5f3f8}}
  .page{{width:min(1100px,calc(100% - 32px));margin:24px auto;background:#fff;border-radius:20px;
        box-shadow:0 18px 50px rgba(58,34,95,.10);padding:36px 44px}}
  h1{{font-size:28px;margin:0 0 4px;letter-spacing:-.02em}}
  .subtitle{{color:var(--muted);margin:0 0 20px;font-size:14px}}
  .kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px}}
  .kpi{{background:linear-gradient(135deg,#fdedf9,#f7f2ff);border-radius:14px;padding:14px 16px}}
  .kpi-label{{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--deep);font-weight:700}}
  .kpi-value{{font-size:28px;font-weight:800;color:var(--ink);margin:2px 0}}
  .kpi-sub{{font-size:12px;color:var(--muted)}}
  .charts{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
  .charts img{{width:100%}}
  h2{{font-size:16px;color:var(--deep);margin:22px 0 10px}}
  ul.hallazgos{{margin:0;padding-left:18px;font-size:13.5px;line-height:1.7}}
  footer{{margin-top:22px;font-size:11px;color:var(--muted);text-align:center}}
</style>
</head>
<body>
<div class="page">
  <h1>Humanos vs. IA — Desempeño en gestiones de cobranza</h1>
  <p class="subtitle">Creceré AI · Prueba Técnica Data Scientist / Data Analyst Junior</p>

  <div class="kpis">{kpi_cards(dataset, resultados)}</div>

  <div class="charts">
    <img src="data:image/png;base64,{img_desempeno}" />
    <img src="data:image/png;base64,{img_conducta}" />
  </div>

  <h2>Hallazgos clave</h2>
  <ul class="hallazgos">{hallazgos_html}</ul>

  <footer>Creceré AI · Technical Challenge · Reporte generado automáticamente desde data/features/</footer>
</div>
</body>
</html>"""

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Reporte guardado en {OUT_PATH}")


if __name__ == "__main__":
    main()

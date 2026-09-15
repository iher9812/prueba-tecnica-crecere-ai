"""
Fase 5 — Genera el reporte HTML ejecutivo (entregable 01).

Lee data/features/dataset.csv y data/features/resultados_tests.csv, genera
gráficos y arma un único HTML autocontenido (máx. 2 páginas) en
report/reporte_final.html, reutilizando la paleta de marca de Creceré AI.

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
SUCCESS = "#2F8F6B"
DANGER = "#D8476B"

plt.rcParams["font.family"] = "sans-serif"


def fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_val(res, variable, col):
    row = res[res.variable == variable]
    return float(row.iloc[0][col]) if len(row) else None


def bar_comparison(res, specs, title):
    """specs: list of (variable, label) tuples. Valores ya vienen 0-1, se muestran en %."""
    labels = [s[1] for s in specs]
    humano_vals = [get_val(res, s[0], "valor_humano") * 100 for s in specs]
    ia_vals = [get_val(res, s[0], "valor_ia") * 100 for s in specs]

    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    y = list(range(len(labels)))
    height = 0.32
    b1 = ax.barh([i + height / 2 for i in y], humano_vals, height=height, color=MUTED, label="Humanos")
    b2 = ax.barh([i - height / 2 for i in y], ia_vals, height=height, color=PURPLE, label="IA")
    for bars, vals in [(b1, humano_vals), (b2, ia_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2, f"{v:.0f}%",
                     va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([])
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right", ncol=2, bbox_to_anchor=(1, -0.28))
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig_to_base64(fig)


def main():
    df = pd.read_csv(DATASET_PATH)
    res = pd.read_csv(RESULTS_PATH)
    n_humano = int((df.grupo == "humano").sum())
    n_ia = int((df.grupo == "ia").sum())

    img_resultados = bar_comparison(
        res,
        [("compromiso_pago", "Compromiso de pago"),
         ("contactabilidad", "Contactabilidad"),
         ("hubo_negociacion", "Hubo negociación")],
        "Resultados de la gestión",
    )
    img_calidad = bar_comparison(
        res,
        [("contradiccion", "Se contradijo / info incorrecta"),
         ("confusion_resuelta_tasa_bien_si_aplica", "Confusión del cliente, resuelta bien"),
         ("manejo_objeciones_tasa_bueno_si_aplica", "Manejo de objeciones, bueno")],
        "Calidad de la conversación",
    )

    compromiso_h = get_val(res, "compromiso_pago", "valor_humano") * 100
    compromiso_ia = get_val(res, "compromiso_pago", "valor_ia") * 100
    contra_h = get_val(res, "contradiccion", "valor_humano") * 100
    contra_ia = get_val(res, "contradiccion", "valor_ia") * 100
    confusion_h = get_val(res, "confusion_resuelta_tasa_bien_si_aplica", "valor_humano") * 100
    confusion_ia = get_val(res, "confusion_resuelta_tasa_bien_si_aplica", "valor_ia") * 100
    obj_h = get_val(res, "objeciones_count", "valor_humano")
    obj_ia = get_val(res, "objeciones_count", "valor_ia")
    manejo_h = get_val(res, "manejo_objeciones_tasa_bueno_si_aplica", "valor_humano") * 100
    manejo_ia = get_val(res, "manejo_objeciones_tasa_bueno_si_aplica", "valor_ia") * 100

    hallazgos = [
        f"<b>La IA se contradice o da información incorrecta en {contra_ia:.0f}% de sus llamadas, "
        f"frente a {contra_h:.0f}% de los humanos</b> — {contra_ia - contra_h:.0f} puntos porcentuales "
        f"de diferencia (p&lt;0.001). Es la brecha más grande y clara del análisis.",

        f"<b>Los humanos logran un compromiso de pago en {compromiso_h:.0f}% de las llamadas, "
        f"el doble que la IA ({compromiso_ia:.0f}%)</b> (p=0.03) — en el objetivo central de la "
        f"gestión, el agente humano es más efectivo.",

        f"<b>Cuando el cliente se confunde, los humanos resuelven bien esa confusión "
        f"{confusion_h:.0f}% de las veces, frente a {confusion_ia:.0f}% de la IA</b> (p=0.03). "
        f"La IA tiende a no despejar la duda del cliente.",

        f"<b>Los clientes ponen más objeciones a los agentes humanos</b> ({obj_h:.1f} vs. "
        f"{obj_ia:.1f} en promedio, p=0.03), pero los humanos también las manejan mejor "
        f"({manejo_h:.0f}% vs. {manejo_ia:.0f}% calificadas como buen manejo) — más diálogo, "
        f"mejor resuelto.",

        "No hay diferencias significativas en duración de la llamada, claridad percibida, ni en "
        "cuántas veces se repite un dato — la brecha entre humanos e IA está en la calidad del "
        "razonamiento (contradicciones) y en la negociación, no en el estilo de comunicación.",
    ]
    hallazgos_html = "".join(f"<li>{h}</li>" for h in hallazgos)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Humanos vs. IA — Reporte Ejecutivo | Creceré AI</title>
<style>
  :root{{
    --pink:{PINK}; --purple:{PURPLE}; --deep:{DEEP}; --ink:{INK}; --muted:{MUTED};
    --bg-soft:#FDEDF9; --line:#E9E4EF; --success:{SUCCESS}; --danger:{DANGER};
  }}
  *{{box-sizing:border-box}}
  body{{
    margin:0; font-family:Inter, ui-sans-serif, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    color:var(--ink); background:#f5f3f8;
  }}
  .page{{
    width:min(1080px, calc(100% - 32px)); margin:24px auto; background:#fff;
    border:1px solid #eeeaf2; border-radius:22px;
    box-shadow:0 18px 50px rgba(58,34,95,.10); padding:40px 48px 32px;
  }}
  .eyebrow{{
    display:inline-flex; align-items:center; gap:8px; padding:6px 12px; border-radius:999px;
    background:var(--bg-soft); color:var(--deep); font-size:11px; font-weight:800;
    letter-spacing:.07em; text-transform:uppercase;
  }}
  h1{{font-size:26px; margin:14px 0 4px; letter-spacing:-.02em; line-height:1.15}}
  .subtitle{{color:var(--muted); margin:0 0 22px; font-size:13.5px}}
  .kpis{{display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:26px}}
  .kpi{{
    background:linear-gradient(135deg,#fdedf9,#f7f2ff); border-radius:14px; padding:14px 16px;
    border:1px solid var(--line);
  }}
  .kpi-label{{font-size:10.5px; text-transform:uppercase; letter-spacing:.05em; color:var(--deep); font-weight:800}}
  .kpi-value{{font-size:26px; font-weight:800; color:var(--ink); margin:3px 0}}
  .kpi-sub{{font-size:11.5px; color:var(--muted)}}
  .charts{{display:grid; grid-template-columns:1fr 1fr; gap:22px; margin-bottom:24px; align-items:start}}
  .charts img{{width:100%; display:block}}
  h2{{font-size:15px; color:var(--deep); margin:0 0 12px; letter-spacing:-.01em}}
  ul.hallazgos{{margin:0; padding-left:20px; font-size:13.5px; line-height:1.75; color:var(--ink)}}
  ul.hallazgos li{{margin-bottom:8px}}
  footer{{
    margin-top:26px; padding-top:16px; border-top:1px solid var(--line);
    font-size:10.5px; color:var(--muted); text-align:center;
  }}
</style>
</head>
<body>
<div class="page">
  <span class="eyebrow">Creceré AI · Prueba Técnica</span>
  <h1>Humanos vs. IA: desempeño en gestiones de cobranza</h1>
  <p class="subtitle">
    Comparación de {n_humano} llamadas de agentes humanos y {n_ia} de un agente de IA, en las
    mismas campañas. Preguntas, hipótesis, código y metodología completa en el
    <a href="../README.md" style="color:var(--purple)">repositorio</a>.
  </p>

  <div class="kpis">
    <div class="kpi">
      <div class="kpi-label">Compromiso de pago</div>
      <div class="kpi-value">{compromiso_h:.0f}% <span style="font-size:14px;color:var(--muted)">vs {compromiso_ia:.0f}%</span></div>
      <div class="kpi-sub">Humanos vs. IA</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Contradicciones de la IA</div>
      <div class="kpi-value">{contra_ia:.0f}% <span style="font-size:14px;color:var(--muted)">vs {contra_h:.0f}%</span></div>
      <div class="kpi-sub">IA vs. Humanos</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Confusión resuelta bien</div>
      <div class="kpi-value">{confusion_h:.0f}% <span style="font-size:14px;color:var(--muted)">vs {confusion_ia:.0f}%</span></div>
      <div class="kpi-sub">Humanos vs. IA</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Llamadas analizadas</div>
      <div class="kpi-value">{n_humano + n_ia}</div>
      <div class="kpi-sub">{n_humano} humanas + {n_ia} IA</div>
    </div>
  </div>

  <div class="charts">
    <div><img src="data:image/png;base64,{img_resultados}" /></div>
    <div><img src="data:image/png;base64,{img_calidad}" /></div>
  </div>

  <h2>Hallazgos clave</h2>
  <ul class="hallazgos">{hallazgos_html}</ul>

  <footer>
    Creceré AI · Technical Challenge · Data Scientist / Data Analyst Junior · Tests estadísticos:
    Mann-Whitney U (variables continuas) y Fisher exacto / Chi-cuadrado (variables categóricas), α=0.05
  </footer>
</div>
</body>
</html>"""

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Reporte guardado en {OUT_PATH}")


if __name__ == "__main__":
    main()

"""
Fase 5 — Genera el reporte HTML ejecutivo (entregable 01).

Lee data/features/dataset.csv y data/features/resultados_tests.csv, genera
gráficos y arma un único HTML autocontenido en report/reporte_final.html.

El reporte replica la identidad visual y la estructura numerada 1-4 del brief
de Creceré AI, de modo que el evaluador vea su propio esquema respondido
punto por punto. El logo se embebe desde report/assets/logo_crecere.b64.

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
LOGO_PATH = ROOT / "report" / "assets" / "logo_crecere.b64"
OUT_PATH = ROOT / "report" / "reporte_final.html"

PINK = "#F26ECF"
PURPLE = "#7047EB"
DEEP = "#4E2AAE"
INK = "#202020"
MUTED = "#757275"

plt.rcParams["font.family"] = "sans-serif"

# Paleta, tipografía y componentes tomados del brief. Tamaños reducidos para
# que el reporte completo quepa en dos páginas A4.
CSS = """
  :root{
    --pink:#F26ECF; --purple:#7047EB; --deep:#4E2AAE; --pink-soft:#F9BCE8;
    --bg-soft:#FDEDF9; --ink:#202020; --muted:#757275; --white:#FFFFFF; --line:#E9E4EF;
  }
  *{box-sizing:border-box}
  body{
    margin:0; background:#f5f3f8; color:var(--ink); line-height:1.45;
    font-family:Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  }
  .page{
    width:min(1080px, calc(100% - 32px)); margin:24px auto; background:var(--white);
    border:1px solid #eeeaf2; border-radius:26px; box-shadow:0 18px 50px rgba(58,34,95,.10);
    overflow:hidden;
  }
  .hero{
    padding:30px 44px 24px; border-bottom:1px solid var(--line);
    background:
      radial-gradient(circle at 88% 12%, rgba(242,110,207,.22), transparent 26%),
      radial-gradient(circle at 10% 90%, rgba(112,71,235,.14), transparent 30%),
      linear-gradient(135deg, #ffffff 0%, #fbf8ff 100%);
  }
  .brand-row{display:flex; align-items:center; justify-content:space-between; gap:20px; margin-bottom:10px}
  .brand-logo{width:150px; max-width:38vw; height:auto; display:block}
  .eyebrow{
    display:inline-flex; align-items:center; gap:8px; padding:7px 12px; border-radius:999px;
    background:var(--bg-soft); color:var(--deep); font-size:11px; font-weight:800;
    letter-spacing:.07em; text-transform:uppercase;
  }
  h1{margin:8px 0 6px; font-size:28px; line-height:1.08; letter-spacing:-.04em; max-width:860px}
  .hero p{margin:0; max-width:820px; font-size:13.5px; color:var(--muted)}
  .hero-grid{margin-top:16px; display:grid; grid-template-columns:repeat(3,1fr); gap:12px}
  .metric{
    border:1px solid var(--line); border-radius:16px; padding:12px 14px;
    background:rgba(255,255,255,.84);
  }
  .metric strong{display:block; font-size:22px; letter-spacing:-.03em; line-height:1.1}
  .metric span{display:block; margin-top:3px; color:var(--muted); font-size:12px}

  .content{padding:24px 44px 28px}
  .section{margin-top:22px}
  .section:first-child{margin-top:0}
  .section-head{display:flex; align-items:center; gap:12px; margin-bottom:10px}
  .num{
    min-width:30px; height:30px; border-radius:9px; display:flex; align-items:center;
    justify-content:center; color:white; font-weight:800; font-size:13px;
    background:linear-gradient(135deg,var(--purple),var(--deep));
    box-shadow:0 8px 18px rgba(112,71,235,.2);
  }
  h2{margin:0; font-size:18px; letter-spacing:-.02em}
  h3{margin:0 0 5px; font-size:12.5px; letter-spacing:-.01em; color:var(--deep)}
  .lead{color:var(--muted); margin:0 0 10px; font-size:12.5px}

  .cards{display:grid; grid-template-columns:repeat(2,1fr); gap:10px}
  .cards-3{grid-template-columns:repeat(3,1fr)}
  .card{border:1px solid var(--line); border-radius:14px; padding:12px 14px; background:#fff}
  .card p{margin:0; color:#39363b; font-size:12.5px; line-height:1.5}

  .pill-wrap{display:flex; flex-wrap:wrap; gap:6px; margin:0 0 12px}
  .pill{
    padding:5px 9px; border-radius:999px; background:#f7f4fb; border:1px solid var(--line);
    font-size:11px; color:#4e4951;
  }
  .pill.nueva{background:var(--bg-soft); border-color:var(--pink-soft); color:var(--deep); font-weight:700}

  .highlight{
    margin-top:12px; padding:12px 16px; border-radius:14px; border:1px solid rgba(112,71,235,.18);
    background:linear-gradient(135deg, rgba(112,71,235,.07), rgba(242,110,207,.07));
    font-size:12.5px; line-height:1.55;
  }
  .highlight strong{color:var(--deep)}
  .quote{
    margin:0 0 10px; padding:12px 16px; border-left:4px solid var(--pink); background:#fff8fd;
    border-radius:0 12px 12px 0; font-weight:700; font-size:13.5px; line-height:1.5;
  }

  .charts{display:grid; grid-template-columns:1fr 1fr; gap:18px; align-items:start}
  .charts img{width:100%; display:block}

  .dos-col{display:grid; grid-template-columns:minmax(220px,0.85fr) 1.6fr; gap:22px; align-items:start}
  table.desc{width:100%; border-collapse:collapse; font-size:11.5px}
  table.desc th{
    text-align:right; color:var(--muted); font-weight:700; padding:4px 8px;
    border-bottom:1px solid var(--line); font-size:10px; text-transform:uppercase; letter-spacing:.04em;
  }
  table.desc th:first-child{text-align:left}
  table.desc td{padding:5px 8px; border-bottom:1px solid #f4f1f8; text-align:right}
  table.desc td:first-child{text-align:left; color:var(--muted)}
  table.desc td:not(:first-child){font-variant-numeric:tabular-nums; font-weight:600; white-space:nowrap}

  .orientadoras{display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-top:14px}
  .orientadora{
    border:1px solid var(--line); border-radius:14px; padding:12px 14px;
    background:linear-gradient(180deg,#fff 0%,#fdfbff 100%);
  }
  .orientadora .q{font-size:11px; color:var(--muted); margin:0 0 6px; line-height:1.4}
  .orientadora .big{
    font-size:20px; font-weight:800; letter-spacing:-.02em; line-height:1.15;
    font-variant-numeric:tabular-nums; margin:0;
  }
  .orientadora .big small{font-size:11.5px; color:var(--muted); font-weight:600; letter-spacing:0}
  .orientadora .sub{font-size:11px; color:#4e4951; margin:5px 0 0; line-height:1.45}

  ul.hallazgos{margin:0; padding-left:18px; font-size:12.5px; line-height:1.6}
  ul.hallazgos li{margin-bottom:7px}

  .nota{
    margin-top:18px; padding:11px 15px; border-radius:12px; background:var(--bg-soft);
    border-left:3px solid var(--pink); font-size:11px; line-height:1.55; color:#4a4450;
  }
  .footer{
    padding:14px 44px 20px; border-top:1px solid var(--line); display:flex;
    justify-content:space-between; gap:18px; color:var(--muted); font-size:11px;
  }

  @media screen and (max-width:800px){
    .hero,.content{padding-left:20px; padding-right:20px}
    .hero-grid,.cards,.cards-3,.charts,.dos-col,.orientadoras{grid-template-columns:1fr}
    .brand-row{align-items:flex-start; flex-direction:column-reverse}
  }
  @media print{
    @page{size:A4; margin:9mm}
    body{background:white}
    .page{width:100%; margin:0; border:none; border-radius:0; box-shadow:none}
    .hero{padding:12px 22px 10px}
    .hero-grid{margin-top:12px}
    .hero p{font-size:12.5px}
    .content{padding:10px 22px 12px}
    .footer{padding:8px 22px 0}
    .section{margin-top:8px}
    .cards{gap:8px}
    .pill-wrap{margin-bottom:9px}
    .orientadoras{margin-top:10px}
    .quote{margin-bottom:8px; padding:10px 14px}
    .nota{margin-top:12px; padding:9px 13px}
    .card,.orientadora,.quote,.nota,.highlight,.metric,.charts img{break-inside:avoid}
    .section-head{break-after:avoid}
    h1{font-size:22px}
    a{color:inherit}
  }
"""


def fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_val(res, variable, col):
    row = res[res.variable == variable]
    return float(row.iloc[0][col]) if len(row) else None


def tasa(df, grupo, col) -> float:
    return df.loc[df.grupo == grupo, col].astype(bool).mean() * 100


def media(df, grupo, col) -> float:
    return df.loc[df.grupo == grupo, col].mean()


def bar_comparison(res, specs, title):
    """specs: list of (variable, label). Valores 0-1 mostrados en %."""
    labels = [s[1] for s in specs]
    humano_vals = [get_val(res, s[0], "valor_humano") * 100 for s in specs]
    ia_vals = [get_val(res, s[0], "valor_ia") * 100 for s in specs]

    fig, ax = plt.subplots(figsize=(6.4, 2.5))
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
    ax.legend(frameon=False, fontsize=9, loc="lower right", ncol=2, bbox_to_anchor=(1, -0.3))
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig_to_base64(fig)


def tabla_descriptiva(df) -> str:
    filas = [
        ("Duración mediana", "duracion_seg", lambda s: f"{s.median():.0f} s"),
        ("Turnos por llamada (mediana)", "num_turnos", lambda s: f"{s.median():.0f}"),
        ("Tiempo de habla del agente", "proporcion_habla_agente", lambda s: f"{s.mean():.0%}"),
        ("Ritmo del agente", "palabras_por_minuto_agente", lambda s: f"{s.mean():.0f} pal/min"),
        ("Objeciones por llamada", "objeciones_count", lambda s: f"{s.mean():.1f}"),
        ("Repitió un dato o pregunta", "dato_repetido", lambda s: f"{s.astype(bool).mean():.0%}"),
        ("Veces que lo repitió (media)", "veces_repetido", lambda s: f"{s.mean():.1f}"),
        ("Claridad percibida (1-5)", "claridad", lambda s: f"{s.mean():.1f}"),
    ]
    cuerpo = ""
    for etiqueta, col, fmt in filas:
        h = fmt(df.loc[df.grupo == "humano", col].dropna())
        ia = fmt(df.loc[df.grupo == "ia", col].dropna())
        cuerpo += f"<tr><td>{etiqueta}</td><td>{h}</td><td>{ia}</td></tr>"
    return cuerpo


def main():
    df = pd.read_csv(DATASET_PATH)
    res = pd.read_csv(RESULTS_PATH)
    logo_b64 = LOGO_PATH.read_text(encoding="utf-8").strip()

    n_humano = int((df.grupo == "humano").sum())
    n_ia = int((df.grupo == "ia").sum())
    n_variables = len(df.columns) - 2
    n_tests = len(res)
    horas_audio = f"{df.duracion_seg.sum() / 3600:.1f}".replace(".", ",")

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

    compromiso_h, compromiso_ia = tasa(df, "humano", "compromiso_pago"), tasa(df, "ia", "compromiso_pago")
    contra_h, contra_ia = tasa(df, "humano", "contradiccion"), tasa(df, "ia", "contradiccion")
    contacta_h, contacta_ia = tasa(df, "humano", "contactabilidad"), tasa(df, "ia", "contactabilidad")
    negocia_h, negocia_ia = tasa(df, "humano", "hubo_negociacion"), tasa(df, "ia", "hubo_negociacion")
    confusion_h = get_val(res, "confusion_resuelta_tasa_bien_si_aplica", "valor_humano") * 100
    confusion_ia = get_val(res, "confusion_resuelta_tasa_bien_si_aplica", "valor_ia") * 100

    # Las tres preguntas orientadoras del análisis, con su cifra.
    rep_h, rep_ia = tasa(df, "humano", "dato_repetido"), tasa(df, "ia", "dato_repetido")
    vrep_h, vrep_ia = media(df, "humano", "veces_repetido"), media(df, "ia", "veces_repetido")
    dist = pd.crosstab(df.grupo, df.confusion_resuelta, normalize="index") * 100
    conf_h = {k: dist.loc["humano"].get(k, 0) for k in ("bien", "mal", "no_aplica")}
    conf_ia = {k: dist.loc["ia"].get(k, 0) for k in ("bien", "mal", "no_aplica")}

    hallazgos = [
        f"<b>Los humanos logran un compromiso de pago en {compromiso_h:.0f}% de las llamadas, "
        f"el doble que la IA ({compromiso_ia:.0f}%)</b> (p=0.03). En el objetivo central de la "
        f"gestión, el agente humano convierte más.",

        f"<b>Cuando el cliente se confunde, los humanos lo resuelven bien {confusion_h:.0f}% de "
        f"las veces, frente a {confusion_ia:.0f}% de la IA</b> (p=0.02). La confusión aparece por "
        f"igual en ambos; lo que cambia es quién la repara.",

        f"<b>La IA ofrece alternativas más seguido ({negocia_ia:.0f}% vs. {negocia_h:.0f}%) pero "
        f"sostiene menos la llamada</b>: contactabilidad {contacta_ia:.0f}% frente a "
        f"{contacta_h:.0f}% (p=0.008). Negocia más y cierra menos.",

        "<b>El estilo de comunicación no explica la brecha.</b> Duración, ritmo, tiempo hablado, "
        "claridad y repetición de datos no difieren de forma significativa. La diferencia está en "
        "la consistencia del discurso y en la reparación conversacional.",
    ]
    hallazgos_html = "".join(f"<li>{h}</li>" for h in hallazgos)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Humanos vs. IA — Reporte Ejecutivo | Creceré AI</title>
<style>{CSS}</style>
</head>
<body>
<main class="page">

  <header class="hero">
    <div class="brand-row">
      <div class="eyebrow">Creceré AI · Prueba técnica · Reporte final</div>
      <img class="brand-logo" src="data:image/png;base64,{logo_b64}" alt="Creceré AI" />
    </div>
    <h1>Humanos vs. IA en gestiones de cobranza</h1>
    <p>
      {n_humano} llamadas de agentes humanos y {n_ia} de un agente de IA, de las mismas campañas.
      Transcritas y anonimizadas en local; cada conclusión está sustentada en variables y tests.
    </p>
    <div class="hero-grid">
      <div class="metric"><strong>{n_humano + n_ia} llamadas</strong><span>{n_humano} humanas + {n_ia} IA · {horas_audio} h de audio</span></div>
      <div class="metric"><strong>{n_variables} variables</strong><span>8 mecánicas del audio + 13 de juicio</span></div>
      <div class="metric"><strong>{n_tests} tests</strong><span>Mann-Whitney U · Fisher exacto · FDR</span></div>
    </div>
  </header>

  <section class="content">

    <div class="section">
      <div class="section-head"><div class="num">1</div><h2>Preguntas, hipótesis y decisiones analíticas</h2></div>
      <div class="cards">
        <div class="card"><h3>¿Qué quiero entender?</h3><p>Si la IA iguala a los humanos en <b>resultado</b> (compromiso de pago) y en <b>conducta</b>: contradicción, repetición de datos y manejo de la confusión del cliente.</p></div>
        <div class="card"><h3>¿Qué variables construí?</h3><p>8 mecánicas del audio (duración, turnos, ritmo) y 13 de juicio extraídas por un LLM sobre transcripciones <b>anonimizadas en local</b>, con esquema JSON estricto.</p></div>
        <div class="card"><h3>¿Qué hipótesis contrasté?</h3><p>H1 IA igual de efectiva → <b>rechazada</b>. H2 IA se contradice menos → <b>rechazada</b>. H3 IA repite más → no confirmada. H4 humanos reparan mejor la confusión → <b>confirmada</b>.</p></div>
        <div class="card"><h3>¿Cómo las probé?</h3><p>Mann-Whitney U (continuas) y Fisher exacto (binarias) sobre 50 vs. 50, corrigiendo por {n_tests} comparaciones múltiples con Benjamini-Hochberg.</p></div>
      </div>
    </div>

    <div class="section">
      <div class="section-head"><div class="num">2</div><h2>Construcción de datos y estadística descriptiva</h2></div>
      <div class="pill-wrap">
        <span class="pill">Contactabilidad</span><span class="pill">Compromiso de pago</span><span class="pill">Negociación</span>
        <span class="pill">Duración</span><span class="pill">Objeciones</span><span class="pill">Manejo de objeciones</span>
        <span class="pill">Claridad</span><span class="pill">Resultado final</span>
        <span class="pill nueva">Contradicción</span><span class="pill nueva">Dato repetido (conteo)</span>
        <span class="pill nueva">Confusión del cliente y su resolución</span>
      </div>
      <div class="dos-col">
        <table class="desc">
          <thead><tr><th></th><th>Humanos</th><th>IA</th></tr></thead>
          <tbody>{tabla_descriptiva(df)}</tbody>
        </table>
        <div class="highlight">
          <strong>Por qué todo se procesó en local.</strong> El censurado de los audios era
          incompleto: sobrevivían nombres de clientes en conversaciones de cobranza — identidad
          unida a situación de mora, dato personal sensible. Por eso transcripción y separación de
          voces corrieron en el equipo, sin nube, y un anonimizador de reglas —sin IA, sin red—
          sustituyó 198 identificadores por seudónimos. Solo esa versión alimentó al LLM. Ningún
          dato identificable salió de la máquina.
        </div>
      </div>
    </div>

    <div class="section">
      <div class="section-head"><div class="num">3</div><h2>Comparación Humanos vs. IA</h2></div>
      <div class="charts">
        <div><img src="data:image/png;base64,{img_resultados}" alt="Resultados de la gestión" /></div>
        <div><img src="data:image/png;base64,{img_calidad}" alt="Calidad de la conversación" /></div>
      </div>
      <div class="orientadoras">
        <div class="orientadora">
          <p class="q">¿Se repitió una pregunta o dato más de una vez?</p>
          <p class="big">{vrep_h:.1f} <small>vs</small> {vrep_ia:.1f} <small>veces</small></p>
          <p class="sub">Humanos vs. IA, promedio por llamada. Ocurre en {rep_h:.0f}% y {rep_ia:.0f}% de las llamadas. Sin diferencia significativa.</p>
        </div>
        <div class="orientadora">
          <p class="q">¿El agente se contradijo o dio información incorrecta?</p>
          <p class="big">{contra_ia:.0f}% <small>IA</small> · {contra_h:.0f}% <small>humanos</small></p>
          <p class="sub">{contra_ia - contra_h:.0f} puntos porcentuales de diferencia (p=0.009). La brecha de conducta más marcada del análisis.</p>
        </div>
        <div class="orientadora">
          <p class="q">¿Hubo confusión del cliente y cómo se resolvió?</p>
          <p class="big">{conf_h['bien']:.0f}% <small>bien</small> · {conf_h['mal']:.0f}% <small>mal</small> · {conf_h['no_aplica']:.0f}% <small>n/a</small></p>
          <p class="sub">Humanos. En la IA: {conf_ia['bien']:.0f}% bien · {conf_ia['mal']:.0f}% mal · {conf_ia['no_aplica']:.0f}% n/a. Cuando hubo confusión, la resolvió bien el {confusion_h:.0f}% de humanos y el {confusion_ia:.0f}% de la IA.</p>
        </div>
      </div>
    </div>

    <div class="section">
      <div class="section-head"><div class="num">4</div><h2>Hallazgos clave</h2></div>
      <div class="quote">La IA se contradice o da información incorrecta en el {contra_ia:.0f}% de sus llamadas, frente al {contra_h:.0f}% de los humanos: una diferencia de {contra_ia - contra_h:.0f} puntos porcentuales.</div>
      <ul class="hallazgos">{hallazgos_html}</ul>
      <div class="nota">
        <b>Cómo leer estas cifras.</b> Los cuatro hallazgos principales son significativos
        individualmente (p&lt;0.05) y apuntan en la misma dirección, pero se contrastaron
        {n_tests} variables sobre la misma muestra: al corregir por comparaciones múltiples ninguno
        alcanza el umbral por sí solo (mejor p ajustada = 0.076). Con {n_humano} llamadas por grupo, la
        lectura correcta es <b>señales convergentes que justifican decisión y seguimiento, no
        hechos establecidos</b>.
      </div>
    </div>

  </section>

  <footer class="footer">
    <span>Creceré AI · Technical Challenge</span>
    <span>Data Scientist / Data Analyst Junior · Metodología, código y datos agregados en el repositorio</span>
  </footer>
</main>
</body>
</html>"""

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Reporte guardado en {OUT_PATH}")


if __name__ == "__main__":
    main()

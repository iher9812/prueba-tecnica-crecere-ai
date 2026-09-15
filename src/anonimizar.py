"""
Anonimización mecánica de transcripciones — sin IA, sin red, 100% determinista.

Reemplaza identificadores directos (nombres, apellidos, documentos, teléfonos,
números de convenio) por seudónimos consistentes dentro de cada llamada.

GARANTÍA DE NO EXPOSICIÓN: este script nunca imprime un valor detectado. A
consola solo salen conteos. Los valores en contexto se escriben en reportes
HTML locales (excluidos de git) para revisión humana. Es lo que permite
auditar la calidad de la redacción sin exponer los datos a ningún servicio
ni a ningún modelo.

Tres capas de detección que se cubren mutuamente:
  (a) patrón de tratamiento    — señor/señora/don/doña + token
  (b) patrón de autopresentación — "mi nombre es X", "le habla X", ...
  (c) diccionario de nombres   — data/diccionarios/nombres_es.txt

Uso:
    python src/anonimizar.py
"""
import json
import re
import unicodedata
from collections import Counter
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTRADA = ROOT / "data" / "transcripts_crudas"
SALIDA = ROOT / "data" / "transcripts_anonimizado"
AUDITORIA_DIR = ROOT / "data" / "anonimizacion"
DICCIONARIO = ROOT / "data" / "diccionarios" / "nombres_es.txt"

PALABRA = r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{3,}"

PATRON_TRATAMIENTO = re.compile(
    rf"\b(?:se[ñn]or(?:a|ita)?|sr|sra|srta|don|do[ñn]a)\s+({PALABRA})", re.IGNORECASE
)
PATRON_PRESENTACION = re.compile(
    rf"\b(?:mi nombre es|me llamo|le habla|habla con|hablando con|hablo con|"
    rf"comunico con|comunicarme con|a nombre de|pregunto por|busco a)\s+({PALABRA})",
    re.IGNORECASE,
)
PATRON_CELULAR = re.compile(r"\b3\d{9}\b")
PATRON_DOCUMENTO = re.compile(r"\b\d{6,12}\b")
PATRON_REFERENCIA = re.compile(
    r"\b(?:convenio|referencia|c[oó]digo)\s+(?:n[uú]mero\s+)?([A-Za-z0-9\-]{3,})",
    re.IGNORECASE,
)

# Whisper transcribe los números en palabras, así que una cédula dictada en voz
# alta ("uno cero nueve dos siete...") nunca aparece como dígitos. Cinco o más
# dígitos-palabra seguidos son casi siempre un documento o un teléfono.
DIGITO_PALABRA = r"(?:cero|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve)"
PATRON_DICTADO = re.compile(rf"\b(?:{DIGITO_PALABRA}\s+){{4,}}{DIGITO_PALABRA}\b", re.IGNORECASE)

# Palabras que legítimamente siguen a un tratamiento y NO son nombres.
# Sin esta lista, "señor presidente" o "señora que" se redactarían como nombre.
STOPLIST = {
    "que", "del", "las", "los", "una", "uno", "con", "por", "para", "sin", "sus",
    "esta", "este", "esto", "esa", "ese", "eso", "aqui", "ahi", "alli", "ya",
    "mas", "muy", "bien", "pues", "entonces", "ahora", "hoy", "manana", "ayer",
    "usted", "ustedes", "senor", "senora", "senorita", "don", "dona",
    "presidente", "cliente", "usuario", "asesor", "asesora", "gerente",
    "titular", "deudor", "deudora", "abogado", "abogada", "doctor", "doctora",
    "juez", "notario", "contador", "jefe", "director", "coordinador",
    "supervisor", "apoderado", "representante", "propietario", "arrendatario",
    "codeudor", "fiador", "beneficiario", "tercero", "familiar",
    "esposo", "esposa", "hijo", "hija", "padre", "madre", "hermano", "hermana",
    "tiene", "puede", "esta", "estaba", "seria", "fue", "era", "hay",
    "porque", "cuando", "donde", "como", "cual", "quien", "todo", "toda",
    "nos", "les", "mio", "mia", "suyo", "suya", "algo", "nada", "solo",
    "banco", "entidad", "empresa", "compania", "obligacion", "cuenta",
    "gracias", "buenos", "buenas", "dias", "tardes", "noches", "favor",
    "acuerdo", "pago", "pagos", "cuota", "cuotas", "saldo", "deuda", "credito",
}

# Palabras frecuentes del español, para el detector de residuos: un token
# capitalizado a mitad de frase que no esté aquí es candidato a nombre no captado.
COMUNES = STOPLIST | {
    "el", "la", "de", "en", "es", "un", "se", "no", "si", "al", "lo", "le",
    "su", "mi", "me", "te", "ha", "he", "va", "voy", "ver", "ser", "dar",
    "pero", "tambien", "desde", "hasta", "sobre", "entre", "menos", "cada",
    "senor", "vale", "listo", "claro", "perfecto", "ok", "okey", "alo",
    "efecty", "bancolombia", "davivienda", "nequi", "daviplata",
}


def normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in sin_tildes if not unicodedata.combining(c))


def cargar_diccionario() -> set:
    tokens = set()
    for linea in DICCIONARIO.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#"):
            tokens.add(normalizar(linea))
    return tokens


def detectar(texto: str, diccionario: set) -> list:
    """Devuelve [(inicio, fin, superficie, tipo, regla)] sin solapamientos."""
    hallazgos = []

    for patron, regla in [(PATRON_TRATAMIENTO, "tratamiento"),
                          (PATRON_PRESENTACION, "presentacion")]:
        for m in patron.finditer(texto):
            superficie = m.group(1)
            if normalizar(superficie) in STOPLIST:
                continue
            hallazgos.append((m.start(1), m.end(1), superficie, "NOMBRE", regla))

    for m in re.finditer(rf"\b{PALABRA}\b", texto):
        if normalizar(m.group(0)) in diccionario:
            hallazgos.append((m.start(), m.end(), m.group(0), "NOMBRE", "diccionario"))

    for m in PATRON_CELULAR.finditer(texto):
        hallazgos.append((m.start(), m.end(), m.group(0), "TELEFONO", "celular"))
    for m in PATRON_DOCUMENTO.finditer(texto):
        hallazgos.append((m.start(), m.end(), m.group(0), "DOCUMENTO", "documento"))
    for m in PATRON_DICTADO.finditer(texto):
        hallazgos.append((m.start(), m.end(), m.group(0), "DOCUMENTO", "dictado"))
    for m in PATRON_REFERENCIA.finditer(texto):
        hallazgos.append((m.start(1), m.end(1), m.group(1), "REFERENCIA", "convenio"))

    hallazgos.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    sin_solape, ultimo_fin = [], -1
    for h in hallazgos:
        if h[0] >= ultimo_fin:
            sin_solape.append(h)
            ultimo_fin = h[1]
    return sin_solape


def construir_mapa(hallazgos: list) -> tuple:
    """Asigna un seudónimo por forma normalizada. El mapa vive solo en memoria."""
    asignaciones, variantes, contadores = {}, {}, Counter()
    for _, _, superficie, tipo, _ in hallazgos:
        clave = normalizar(superficie)
        if clave not in asignaciones:
            contadores[tipo] += 1
            asignaciones[clave] = f"[{tipo}_{contadores[tipo]}]"
            variantes[clave] = set()
        variantes[clave].add(superficie)
    return asignaciones, variantes


def aplicar(texto: str, asignaciones: dict, variantes: dict) -> str:
    for clave, formas in variantes.items():
        for forma in sorted(formas, key=len, reverse=True):
            texto = re.sub(rf"\b{re.escape(forma)}\b", asignaciones[clave], texto)
    return texto


def buscar_residuos(turnos: list) -> list:
    """
    Candidatos NO redactados que parecen identificador. Es la medida de recall.

    Se recorre turno por turno y se descartan los tokens en posición inicial de
    turno o de frase: Whisper capitaliza siempre ahí, y contarlos ahogaría la
    señal real bajo cientos de falsas alarmas.
    """
    residuos = []
    for turno in turnos:
        texto = turno["texto"]

        for m in PATRON_TRATAMIENTO.finditer(texto):
            token = m.group(1)
            if not token.startswith("[") and normalizar(token) not in STOPLIST:
                residuos.append((texto, m.start(1), m.end(1), "tras tratamiento"))

        for m in re.finditer(r"\b\d{6,}\b", texto):
            residuos.append((texto, m.start(), m.end(), "dígitos largos"))

        for m in re.finditer(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}\b", texto):
            previo = texto[:m.start()].rstrip()
            if not previo or previo[-1] in ".?!¿¡":
                continue
            if normalizar(m.group(0)) not in COMUNES:
                residuos.append((texto, m.start(), m.end(), "capitalizado"))
    return residuos


def contexto(texto: str, inicio: int, fin: int, margen: int = 45) -> str:
    izq = escape(texto[max(0, inicio - margen):inicio])
    match = escape(texto[inicio:fin])
    der = escape(texto[fin:fin + margen])
    return f"…{izq}<mark>{match}</mark>{der}…"


def escribir_reporte(ruta: Path, titulo: str, subtitulo: str, filas: list, columnas: list):
    encabezado = "".join(f"<th>{c}</th>" for c in columnas)
    cuerpo = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in fila) + "</tr>" for fila in filas)
    ruta.write_text(f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8"><title>{titulo}</title>
<style>
 body{{font-family:Inter,system-ui,Arial,sans-serif;margin:32px;color:#202020;background:#f7f5fa}}
 h1{{font-size:20px;margin:0 0 4px}} p.sub{{color:#757275;font-size:13px;margin:0 0 20px}}
 .aviso{{background:#FDEDF9;border-left:4px solid #F26ECF;padding:10px 14px;
        border-radius:6px;font-size:12.5px;margin-bottom:20px}}
 table{{border-collapse:collapse;width:100%;background:#fff;border-radius:10px;overflow:hidden;
        box-shadow:0 2px 12px rgba(58,34,95,.08)}}
 th{{background:#4E2AAE;color:#fff;text-align:left;padding:9px 12px;font-size:12px}}
 td{{padding:8px 12px;border-top:1px solid #eee;font-size:12.5px;vertical-align:top}}
 mark{{background:#F9BCE8;padding:1px 3px;border-radius:3px;font-weight:600}}
 code{{background:#f0edf7;padding:1px 5px;border-radius:4px;font-size:11.5px}}
</style></head><body>
<h1>{titulo}</h1><p class="sub">{subtitulo}</p>
<div class="aviso"><b>Archivo local.</b> Está excluido de git y no se comparte con
ningún servicio ni modelo. Es el único lugar donde los valores originales aparecen
en contexto, para que puedas verificar la calidad de la redacción tú misma.</div>
<table><thead><tr>{encabezado}</tr></thead><tbody>{cuerpo}</tbody></table>
</body></html>""", encoding="utf-8")


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    AUDITORIA_DIR.mkdir(parents=True, exist_ok=True)
    diccionario = cargar_diccionario()

    archivos = sorted(ENTRADA.glob("*.json"))
    filas_auditoria, filas_residuos = [], []
    por_regla, por_tipo = Counter(), Counter()
    residuos_por_motivo = Counter()
    archivos_con_deteccion = 0

    for ruta in archivos:
        d = json.loads(ruta.read_text(encoding="utf-8"))
        original = d.get("texto_completo", "")

        hallazgos = detectar(original, diccionario)
        asignaciones, variantes = construir_mapa(hallazgos)

        for inicio, fin, superficie, tipo, regla in hallazgos:
            por_regla[regla] += 1
            por_tipo[tipo] += 1
            filas_auditoria.append([
                f"<code>{d['id'][:8]}</code>", d["grupo"], f"<code>{regla}</code>",
                contexto(original, inicio, fin),
                f"<code>{asignaciones[normalizar(superficie)]}</code>",
            ])
        if hallazgos:
            archivos_con_deteccion += 1

        for turno in d["turnos"]:
            turno["texto"] = aplicar(turno["texto"], asignaciones, variantes)
        d["texto_completo"] = "\n".join(f"[{t['rol']}] {t['texto']}" for t in d["turnos"])

        for fuente, inicio, fin, motivo in buscar_residuos(d["turnos"]):
            residuos_por_motivo[motivo] += 1
            filas_residuos.append([
                f"<code>{d['id'][:8]}</code>", motivo, contexto(fuente, inicio, fin),
            ])

        (SALIDA / ruta.name).write_text(
            json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    escribir_reporte(
        AUDITORIA_DIR / "auditoria.html",
        "Auditoría de anonimización",
        f"{sum(por_tipo.values())} identificadores detectados y reemplazados en "
        f"{archivos_con_deteccion} de {len(archivos)} transcripciones.",
        filas_auditoria, ["Llamada", "Grupo", "Regla", "Contexto", "Reemplazo"],
    )
    escribir_reporte(
        AUDITORIA_DIR / "residuos.html",
        "Residuos — posibles identificadores NO capturados",
        f"{len(filas_residuos)} candidatos para revisión. Es la medida de recall: "
        f"si aquí aparecen nombres reales, hay que ajustar las reglas o el diccionario.",
        filas_residuos, ["Llamada", "Motivo", "Contexto"],
    )

    resumen = {
        "archivos_procesados": len(archivos),
        "archivos_con_deteccion": archivos_con_deteccion,
        "detecciones_por_tipo": dict(por_tipo),
        "detecciones_por_regla": dict(por_regla),
        "residuos_por_motivo": dict(residuos_por_motivo),
        "total_detecciones": sum(por_tipo.values()),
        "total_residuos": len(filas_residuos),
    }
    (AUDITORIA_DIR / "resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    print(f"\nTranscripciones anonimizadas -> {SALIDA}")
    print(f"Revisa localmente: {AUDITORIA_DIR / 'auditoria.html'}")
    print(f"Revisa localmente: {AUDITORIA_DIR / 'residuos.html'}")


if __name__ == "__main__":
    main()

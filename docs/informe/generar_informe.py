"""Genera el informe de estado de La MasIA (PDF). Uso: uv run --no-project --with reportlab==4.2.5 --with matplotlib==3.9.2 --with numpy==1.26.4 python docs/informe/generar_informe.py <salida.pdf>"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

OUT = sys.argv[1]
TMP = os.path.dirname(os.path.abspath(__file__))

F = "C:/Windows/Fonts/"
pdfmetrics.registerFont(TTFont("Arial", F + "arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", F + "arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", F + "ariali.ttf"))
pdfmetrics.registerFont(TTFont("Arial-BoldItalic", F + "arialbi.ttf"))
pdfmetrics.registerFont(TTFont("Consolas", F + "consola.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-BoldItalic")

BLUE = colors.HexColor("#1f3b73")
LIGHT = colors.HexColor("#eef2f9")
GREEN = colors.HexColor("#e7f4ea")
RED = colors.HexColor("#fbeaea")
YELLOW = colors.HexColor("#fdf6e3")

ss = getSampleStyleSheet()
base = ParagraphStyle("base", parent=ss["Normal"], fontName="Arial", fontSize=10, leading=14, spaceAfter=5)
small = ParagraphStyle("small", parent=base, fontSize=8.5, leading=11)
cell = ParagraphStyle("cell", parent=base, fontSize=8.5, leading=11, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Arial-Bold", textColor=colors.white)
h1 = ParagraphStyle("h1", parent=base, fontName="Arial-Bold", fontSize=17, leading=21, textColor=BLUE,
                    spaceBefore=4, spaceAfter=10)
h2 = ParagraphStyle("h2", parent=base, fontName="Arial-Bold", fontSize=13, leading=17, textColor=BLUE,
                    spaceBefore=10, spaceAfter=6)
h3 = ParagraphStyle("h3", parent=base, fontName="Arial-Bold", fontSize=11, leading=14, spaceBefore=6, spaceAfter=3)
title = ParagraphStyle("title", parent=base, fontName="Arial-Bold", fontSize=26, leading=32, textColor=BLUE,
                       alignment=TA_CENTER)
subtitle = ParagraphStyle("subtitle", parent=base, fontSize=13, leading=18, alignment=TA_CENTER, textColor=colors.grey)
code = ParagraphStyle("code", parent=base, fontName="Consolas", fontSize=8.5, leading=11, backColor=LIGHT,
                      borderPadding=4, spaceBefore=3, spaceAfter=7)
box_style = ParagraphStyle("box", parent=base, spaceAfter=0)

story = []


def P(text, style=base):
    story.append(Paragraph(text, style))


def H1(text):
    story.append(Paragraph(text, h1))


def H2(text):
    story.append(Paragraph(text, h2))


def H3(text):
    story.append(Paragraph(text, h3))


def bullets(items, style=base):
    story.append(ListFlowable([ListItem(Paragraph(t, style), leftIndent=12, value="circle") for t in items],
                              bulletType="bullet", start="•", leftIndent=14, bulletFontSize=8))
    story.append(Spacer(1, 3))


def numbered(items):
    story.append(ListFlowable([ListItem(Paragraph(t, base), leftIndent=14) for t in items],
                              bulletType="1", leftIndent=16))
    story.append(Spacer(1, 3))


def table(rows, widths, head=True, shade=None):
    data = [[Paragraph(str(c), cellb if (head and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if head else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c2d6")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if head:
        st += [("BACKGROUND", (0, 0), (-1, 0), BLUE)]
    for i in range(1 if head else 0, len(rows)):
        if i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    for (r, color) in (shade or []):
        st.append(("BACKGROUND", (0, r), (-1, r), color))
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 8))


def box(text, color=YELLOW):
    t = Table([[Paragraph(text, box_style)]], colWidths=[17 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), color), ("BOX", (0, 0), (-1, -1), 0.6, BLUE),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(t)
    story.append(Spacer(1, 8))


def figure(path, width_cm, caption):
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(path).getSize()
    story.append(KeepTogether([Image(path, width=width_cm * cm, height=width_cm * cm * h / w),
                               Paragraph("<i>" + caption + "</i>", small)]))
    story.append(Spacer(1, 8))


# ------------------------------------------------------------------------------------ graficos
plt.rcParams.update({"font.family": "Arial", "font.size": 9})


def fig_progress():
    labels = ["Baseline\noficial", "L1 reglas\n+ Viterbi", "L3\nLightGBM", "L3 + toques\nrelativos",
              "L3 apilado", "Apilado\n(GT controlada)"]
    vals = [0.300, 0.326, 0.502, 0.519, 0.542, 0.6245]
    benches = ["v2", "v2", "v2", "v2.1", "v2.1", "v2.1 ctl"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bars = ax.bar(labels, vals, color=["#9aa7c0"] * 2 + ["#4a6fa5"] * 3 + ["#1f3b73"])
    for b, v, bn in zip(bars, vals, benches):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, "%.3f\n(banco %s)" % (v, bn), ha="center", fontsize=7.5)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("spellF1 (banco, partido fuera)")
    ax.set_title("Fase 1: evolución en el banco de 3 partidos (Metrica)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(TMP, "fig_progress.png")
    fig.savefig(p, dpi=180)
    return p


def fig_v3():
    games = ["M1", "M2", "M3", "WMX", "WN1", "WPY", "WOH", "WQQ", "WOY", "WR9"]
    ref = [0.660, 0.633, 0.693, 0.654, 0.584, 0.681, 0.673, 0.615, 0.620, 0.661]
    fam = [0.745, 0.721, 0.750, 0.711, 0.672, 0.746, 0.748, 0.725, 0.734, 0.631]
    import numpy as np
    x = np.arange(len(games))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(x - 0.2, ref, 0.4, label="Referencia (features v3): 0,647", color="#9aa7c0")
    ax.bar(x + 0.2, fam, 0.4, label="Con familias de features: 0,7185", color="#1f3b73")
    ax.set_xticks(x, games)
    ax.set_ylim(0.5, 0.8)
    ax.set_ylabel("spellF1 (partido fuera)")
    ax.set_title("Banco v3 (3 Metrica + 7 DFL): referencia frente a familias de features")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(TMP, "fig_v3.png")
    fig.savefig(p, dpi=180)
    return p


def fig_failures():
    cats = ["Equipo contrario\n(robo no visto)", "Fusionada: posesión\nrival corta", "Fusionada: rival\ny parón",
            "Tapada por D", "Fronteras\ndesplazadas", "Partida en\ntrozos", "Fusionada:\nparón"]
    pct = [42, 23, 14, 10, 5, 3, 3]
    fig, ax = plt.subplots(figsize=(8, 3.0))
    colors_ = ["#b03a2e", "#d35400", "#d35400", "#7f8c8d", "#95a5a6", "#95a5a6", "#95a5a6"]
    bars = ax.barh(cats[::-1], pct[::-1], color=colors_[::-1])
    for b, v in zip(bars, pct[::-1]):
        ax.text(v + 0.5, b.get_y() + b.get_height() / 2, "%d %%" % v, va="center", fontsize=8)
    ax.set_xlabel("% de las posesiones reales perdidas (banco, GT controlada, E020)")
    ax.set_title("Por qué se pierde cada posesión: el 79 % son robos que no vemos")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(TMP, "fig_failures.png")
    fig.savefig(p, dpi=180)
    return p


def fig_metric():
    import numpy as np
    fig, axes = plt.subplots(3, 1, figsize=(8, 2.6), sharex=True)
    t = np.arange(60)
    real = np.where(t < 20, 0, np.where(t < 28, 1, np.where(t < 34, 2, 0)))
    good = real.copy()
    bad = np.where(t < 34, 0, 0)
    bad = np.where((t >= 28) & (t < 34), 2, 0)
    flick = real.copy()
    flick[[8, 12, 44, 50]] = 1
    cmap = {0: "#1f3b73", 1: "#c0392b", 2: "#bdc3c7"}
    for ax, seq, name in zip(axes, (real, bad, flick), ("Real", "Robo no visto", "Parpadeo")):
        for i, v in enumerate(seq):
            ax.axvspan(i, i + 1, color=cmap[v])
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
    axes[-1].set_xticks([])
    axes[0].set_title("Azul = Barça (H), rojo = Athletic (A), gris = balón muerto (D)", fontsize=8)
    fig.tight_layout()
    p = os.path.join(TMP, "fig_metric.png")
    fig.savefig(p, dpi=180)
    return p


FIG_P, FIG_V3, FIG_F, FIG_M = fig_progress(), fig_v3(), fig_failures(), fig_metric()

# ------------------------------------------------------------------------------------ portada
story.append(Spacer(1, 5 * cm))
P("La MasIA", title)
P("More Than A Hack 2027 · Fase 1: posesión por frame", subtitle)
story.append(Spacer(1, 1 * cm))
P("Informe de estado: qué hemos hecho, por qué, dónde fallamos y qué viene", subtitle)
story.append(Spacer(1, 2.5 * cm))
P("24 de octubre de 2026 · Rama <b>banco-v3</b> del repositorio "
  "github.com/ManelCM/FCBCN_HACK_2027 · Preparado por Manel con el agente", ParagraphStyle("c", parent=base, alignment=TA_CENTER))
story.append(Spacer(1, 1 * cm))
box("<b>Para quién es esto.</b> Para ponernos al día los tres. Está escrito para leerse de principio a fin "
    "(unos 30-40 minutos), pero cada sección se entiende sola. Los números salen de "
    "<font face='Consolas'>docs/experimentos.md</font> (identificadores E001-E037) y de los ADR de "
    "<font face='Consolas'>docs/decisiones/</font>. Si algo de aquí contradice esos ficheros, mandan ellos.", LIGHT)
story.append(PageBreak())

# ------------------------------------------------------------------------------------ resumen
H1("1. Resumen en una página")
P("<b>La tarea.</b> Para cada uno de los 199.319 frames del Barça-Athletic hay que decir si el balón es del "
  "Barça (H), del Athletic (A) o está muerto (D). Se puntúa con <b>spellF1</b>, que premia acertar "
  "<b>posesiones enteras</b> (no frames sueltos). El baseline oficial saca 0,243. No hay leaderboard: "
  "solo sabremos la nota al final.")
P("<b>Dónde estamos.</b>")
bullets([
    "Tenemos un modelo que en nuestro banco de validación (10 partidos públicos estropeados para parecerse al "
    "tracking del reto) saca <b>0,7185</b> de spellF1. Hace dos días sacaba 0,6245 en un banco más pequeño, y "
    "la referencia del banco actual era 0,6470.",
    "La gran mejora de esta semana (<b>+0,072</b>, E035) viene de las <b>familias de features pensadas como "
    "las mira un entrenador</b>: balón, jugadores (quién aprieta, quién esprinta, quién se desmarca, el "
    "portero), bloques (cómo de juntos y abiertos están, altura de líneas) y estructura (forma con y sin "
    "balón, cuántos hay por detrás del balón).",
    "Estimación orientativa en el reto: el banco suele ser algo más fácil que el reto (~0,04 con el "
    "baseline). Con el modelo nuevo, esperamos <b>del orden de 0,62-0,68</b>, muy por encima del 0,243 del "
    "baseline. Es una estimación con mucha incertidumbre (±0,05 o más).",
    "<font face='Consolas'>La MasIA.csv</font> existe, es válido y reproducible bit a bit, pero <b>todavía sale "
    "del modelo anterior</b> (0,6245 en el banco pequeño). Hay que regenerarlo con el modelo nuevo.",
])
P("<b>Qué falta (por orden).</b>")
numbered([
    "Ablación por familias (quitar una cada vez) para saber de dónde sale la mejora y descartar que sea un "
    "artefacto del simulador. La primera ablación se cortó al reiniciarse la sesión; hay que relanzarla.",
    "Entrenar el modelo final con los 10 partidos y regenerar <font face='Consolas'>La MasIA.csv</font> "
    "(dos veces, mismo sha) + prueba de humo con los 5 minutos públicos.",
    "Documento técnico definitivo, repositorio público, PR a main aprobada por un compañero, y subir el "
    "formulario antes del 17 de noviembre (congelamos el 14).",
])
P("<b>Qué necesitamos de vosotros.</b> Enviar el email con las preguntas a la organización (Q2/Q4/Q5/Q6); "
  "aceptar la invitación de GitHub y revisar la PR; confirmar los tres la inscripción antes del 10 nov; decidir "
  "entre los tres si el ADR-0004 recoge que el agente ejecute scripts de solo agregados (sección 11).")
story.append(PageBreak())

# ------------------------------------------------------------------------------------ reto
H1("2. El reto y la métrica")
H2("2.1 La tarea")
P("Un partido: FC Barcelona (local, H) - Athletic Club (visitante, A), LaLiga 2024-25, jornada 2. El vídeo "
  "táctico tiene 199.319 frames a 25 fps; se puntúan 158.857 (las dos partes). Por cada frame se entrega una "
  "etiqueta:")
table([["Etiqueta", "Significado"],
       ["H", "Posesión del Barça con el balón vivo"],
       ["A", "Posesión del Athletic con el balón vivo"],
       ["D", "Balón muerto (falta, fuera, gol, descanso de juego...)"],
       ["X", "Sin respuesta; solo la usamos fuera de las partes, donde no cuenta"]], [3, 14])
P("Entregables: <font face='Consolas'>La MasIA.csv</font> (199.319 filas), un repositorio <b>público</b> que "
  "regenere <b>exactamente</b> el mismo fichero con un comando, y un documento técnico corto. Fecha límite: "
  "17 de noviembre. Pasan los 20 mejores a la fase 2.")
H2("2.2 Por qué spellF1 lo cambia todo")
P("spellF1 no cuenta frames: cuenta <b>posesiones</b>. Una posesión es una racha seguida de H o de A; un D en "
  "medio la corta. Cada posesión predicha se empareja con una real si se solapan lo suficiente (IoU, "
  "intersección entre unión) y se promedia la F1 en umbrales de IoU de 0,3 a 0,7. Consecuencias prácticas:")
figure(FIG_M, 15, "Figura 1. Arriba, la realidad: H, luego A (un robo corto), luego un parón y otra vez H. "
                  "En medio, un modelo que no ve el robo: une las dos posesiones del Barça y pierde la del "
                  "Athletic. Abajo, un modelo que 'parpadea': cada parpadeo crea dos posesiones falsas y rompe "
                  "una buena. Las dos cosas se pagan muy caras, aunque el acierto por frame sea alto.")
bullets([
    "<b>No ver un robo cuesta 2-3 posesiones</b>: la del rival desaparece y las dos del otro equipo se funden "
    "en una que ya no empareja con ninguna.",
    "<b>Parpadear es letal</b>: el baseline oficial sin suavizar tiene macroF1 0,65 pero spellF1 0,05.",
    "<b>Llegar 1 segundo tarde a un cambio cuesta ~35 %</b> de la nota, porque hay muchas posesiones cortas.",
    "<b>Detectar bien los D</b> importa: un parón no detectado funde dos posesiones; un D falso parte una.",
    "La organización avisa de que diferencias &lt; 0,04 en el partido completo son ruido.",
])
H2("2.3 Qué cuenta como posesión")
P("Según CHALLENGE.md, la posesión cambia cuando el rival gana el balón <b>controlado</b>. No cuentan un toque "
  "sin control, un desvío que no intercepta ni un duelo aéreo sin control posterior. La GT sale del tracking "
  "corregido de Metrica de forma mecánica, y la referencia práctica son los 5 minutos públicos (17 posesiones, "
  "14,2 s de media, 17,6 % de posesiones de menos de 3 s). Esto fue decisivo (sección 4.3).")
H2("2.4 Las reglas que más nos condicionan")
bullets([
    "<b>Ningún LLM en la nube puede ver los datos del reto ni extractos de ellos</b> (T&amp;C). Nuestro agente "
    "es un LLM en la nube, así que nunca lee los datos: escribe el código, el código se ejecuta en local y "
    "solo vuelven números agregados (ADR-0004, sección 11).",
    "Prohibido etiquetar a mano el resultado; prohibido usar otras fuentes de posesión de este partido.",
    "El fichero tiene que regenerarse <b>idéntico</b> desde el repositorio.",
    "<b>No hay leaderboard</b>: los 5 minutos públicos no sirven para elegir modelos (el propio repo oficial "
    "lo demuestra: con ellos se elige mal hasta el suavizado del baseline).",
])


# ------------------------------------------------------------------------------------ datos
H1("3. Los datos y qué hacemos con cada uno")
table([["Fuente", "Qué es", "Uso actual y por qué"],
       ["ATD táctico", "Posición de los jugadores y del balón en cada frame, automático, sin identidades "
        "(46 slots que se fragmentan)", "<b>La base de todo.</b> Problemas: falta el balón en el 41 % de los "
        "frames puntuados (en ~134 huecos largos) y la trayectoria del balón está muy filtrada (casi no se "
        "ven los toques)."],
       ["Vídeo táctico", "1080p, homografía válida en todos los frames; audio mudo",
        "Probamos un detector de balón (YOLO): <b>descartado</b>, solo acierta el 24 % (E025)."],
       ["Broadcast", "Cámara de tele con su propio tracking; no sincronizada; audio real con comentarista "
        "en inglés", "Alineado con el táctico (un desfase fijo por parte, E026). Como fuente de balón "
        "<b>no sirve</b> (rellena el 7,5 % de los huecos, E023). El silbato no se detecta (E029); el "
        "comentario se transcribe bien pero aporta poco (E033)."],
       ["Smart Tagging", "Eventos automáticos de Metrica (fases, balones parados)",
        "<b>Descartado</b>: +0,008, dentro del ruido, y solo el 13 % de los huecos largos acaba en un balón "
        "parado etiquetado (ADR-0007)."],
       ["GT pública 5 min", "Las etiquetas reales de los primeros 5 minutos",
        "Solo prueba de humo (detectar roturas) y para conocer la forma de la definición de posesión. "
        "<b>Nunca elige modelos.</b>"],
       ["Metrica sample-data", "3 partidos públicos con tracking limpio y eventos",
        "Banco de validación y entrenamiento. Su GT sale de los eventos."],
       ["DFL / Sportec", "7 partidos de la Bundesliga con tracking corregido y flags de posesión por frame",
        "Añadidos el 22-23 oct (banco v3). Su GT sale del tracking, como la del reto."]],
      [3, 5.5, 8.5])
box("<b>Lo más importante de los datos:</b> el balón falta justo en los momentos que deciden la posesión "
    "(choques, áreas llenas, balones aéreos). Por eso el reto no se gana mirando solo al balón, sino a lo "
    "que hacen los 22 jugadores, que es lo que hemos atacado esta semana.")

# ------------------------------------------------------------------------------------ validacion
H1("4. Cómo validamos sin leaderboard")
H2("4.1 El banco")
P("Como no hay leaderboard y la GT completa es privada, construimos un <b>banco de validación</b> con partidos "
  "públicos. Su tracking limpio se <b>estropea a propósito</b> para parecerse al del reto: se borra el balón "
  "en huecos largos (más probables con balón muerto), se suaviza su trayectoria como lo hace el tracker de "
  "Metrica, se parten las trayectorias de los jugadores en 46 slots sin identidad y se añade ruido. "
  "(ADR-0005, ADR-0006, ADR-0009.)")
H2("4.2 Cómo sabemos que el banco se parece al reto")
P("Con dos anclas: (1) el baseline oficial tiene que dar en el banco el mismo orden y casi la misma escala "
  "que los cuatro números que publicó la organización para el partido real; (2) estadísticas agregadas del "
  "ATD real (porcentaje de balón ausente, longitud de los huecos, dinámica del balón).")
table([["Baseline oficial", "raw", "s25", "s50", "s125"],
       ["Reto real (publicado)", "0,049", "0,200", "0,243", "0,224"],
       ["Banco Metrica (GT controlada)", "0,040", "0,194", "0,280", "0,275"],
       ["Banco DFL (mismo simulador, sin tocar)", "0,025", "0,203", "0,265", "0,236"]],
      [6.5, 2.6, 2.6, 2.6, 2.7])
P("Los dos bancos reproducen el orden del reto (s50 &gt; s125 &gt; s25 &gt; raw) y quedan ~0,02-0,04 más "
  "fáciles. Por eso los números del banco sirven para <b>ordenar</b> modelos, no para predecir la nota exacta.")
H2("4.3 La corrección clave: la posesión «controlada» (ADR-0012)")
P("El 22 de octubre descubrimos que el banco medía otra definición: la GT de Metrica cambiaba de equipo en "
  "<b>cada</b> recuperación, aunque durase un segundo (36 % de posesiones de menos de 3 s, frente al 17,6 % de "
  "la GT pública). Aplicamos la regla «un robo solo cuenta si el nuevo equipo conserva el balón al menos T "
  "segundos» (tras un parón siempre cuenta). T se elige por la <b>forma</b> de la distribución, nunca por la "
  "nota: con T = 1,5 s Metrica reproduce la GT pública (17,5 % y 13,7 s). En DFL hace falta T = 2 s (17,5 %).")
box("<b>Por qué importa:</b> el mismo modelo, sin tocarlo, pasó de 0,542 a 0,628. Reentrenarlo con la nueva "
    "definición no aportó (E021): el modelo ya era bueno; lo que estaba mal era el examen. Desde entonces "
    "todo se mide con la GT controlada y los números anteriores no son comparables.", GREEN)
H2("4.4 El protocolo para aceptar cambios (ADR-0008)")
bullets([
    "Siempre <b>dejando un partido fuera</b>: se entrena con los demás y se puntúa el que no se ha visto, "
    "en todas las combinaciones. Los parámetros del Viterbi también se eligen sin mirar el partido retenido.",
    "Un cambio entra solo si sube la nota agregada <b>≥ 0,01</b> y mejora en <b>al menos 7 de 10</b> partidos "
    "(antes, 3 de 3). Así evitamos quedarnos con la variante que tuvo suerte.",
    "Antes de aceptar features nuevas se comprueba que su distribución en el ATD real se parece a la del banco "
    "(<font face='Consolas'>feature_shift</font>), porque una vez unas features de toques funcionaban en el "
    "banco y no en el reto (E012-E013).",
    "Todo se registra en <font face='Consolas'>docs/experimentos.md</font>, también lo que falla.",
])
H2("4.5 El banco v3: de 3 a 10 partidos")
P("Con 3 partidos, una mejora pequeña no superaba el ruido. El 22-23 oct añadimos los 7 partidos de DFL "
  "(descargados abriendo las URLs de figshare en el navegador, que bloquea las descargas automáticas). Su GT "
  "sale de los flags de posesión de su tracking, la misma naturaleza que la del reto, aunque también cambia "
  "de equipo con cada toque (43,7 % de posesiones de menos de 3 s), así que necesita la regla controlada. "
  "La referencia en el banco v3 es <b>0,6470</b> (E030). Detalle útil: entrenar con 9 partidos en vez de 2 ya "
  "sube los de Metrica unos 0,04, así que el modelo final se entrenará con los 10.")


# ------------------------------------------------------------------------------------ modelo
H1("5. El modelo, paso a paso")
P("Es un sistema de tres piezas. La idea: un clasificador da probabilidades por frame y un decodificador las "
  "convierte en posesiones coherentes.")
numbered([
    "<b>Features por frame.</b> Unos 65 números en el modelo actual (más de 150 con las familias nuevas). "
    "Todo es colectivo y sin identidades: cercanía de cada equipo al balón, toques (con un umbral relativo a "
    "cada partido, porque el balón real va filtrado), receptor probable, velocidad colectiva, forma de los "
    "equipos, contexto de los huecos sin balón... Se calculan también en ventanas antes y después de cada "
    "frame (el sistema es offline: ve el partido entero).",
    "<b>Capa 1: LightGBM</b> (127 hojas, 400 rondas) que da P(H), P(A), P(D). Se entrena también con los "
    "equipos intercambiados para no tener sesgo de local, y la predicción se simetriza.",
    "<b>Capa 2 (apilado, ADR-0011)</b>: otro LightGBM que ve las probabilidades de la capa 1 resumidas en "
    "ventanas de 1 a 10 s hacia atrás, centradas y hacia delante. Se entrena con probabilidades fuera de "
    "muestra para no aprender de predicciones demasiado buenas. Elimina saltos sueltos y aporta contexto largo.",
    "<b>Viterbi</b>: busca la secuencia H/A/D más probable de cada parte, cobrando por cada cambio. Como decide "
    "la parte entera de una vez, coloca cada frontera donde mejor encaja la evidencia a los dos lados: no hay "
    "retraso. Es imprescindible: sin Viterbi, el mismo modelo saca 0,21; con él, 0,65-0,72.",
])
P("<b>Reproducibilidad.</b> El modelo entrenado se guarda en el repositorio (no se reentrena al entregar). El "
  "entrenamiento es determinista bit a bit (lo arreglamos el 22 oct: LightGBM elegía el método de "
  "histograma cronometrando la máquina, E024). <font face='Consolas'>make_submission</font> genera el CSV, lo "
  "valida con el validador oficial e imprime su sha256; dos ejecuciones dan el mismo fichero.")
figure(FIG_P, 16, "Figura 2. Evolución en el banco antiguo de 3 partidos. Ojo: los bancos cambian (v2, v2.1, "
                  "GT controlada), así que las barras de grupos distintos no se comparan entre sí; cada mejora "
                  "se midió frente a la anterior en el mismo banco.")


# ------------------------------------------------------------------------------------ cronologia
H1("6. Lo que hemos probado, con números")
P("Cada fila resume un experimento. «Entra» significa que pasó el protocolo y forma parte del modelo.")
table([["Exp.", "Qué", "Resultado", "Decisión y por qué"],
       ["E001-E002", "Banco Metrica + simulador calibrado con el baseline", "Reproduce el orden del reto",
        "Base de toda la validación"],
       ["E003-E004", "L1: reglas + Viterbi (+ Smart Tagging)", "0,337-0,363", "Superado por L3"],
       ["E005-E006", "Diagnóstico del ATD real (agregados)", "41 % sin balón, huecos largos", "Banco v2 con "
        "huecos largos"],
       ["E007-E010", "L3: LightGBM + Viterbi; más capacidad", "0,502 (v2)", "<b>Entra</b> (ADR-0007)"],
       ["E012-E014", "Features de toques: umbral fijo → relativo", "Fijo: espejismo; relativo +0,054",
        "<b>Entra el relativo</b>: el balón real va filtrado (ADR-0009)"],
       ["E015-E017", "Empuje de bloque; aleatorización del dominio", "+0,009; neutro", "Descartados"],
       ["E018-E019", "Apilado (capa 2)", "0,542 (+0,029, 3/3)", "<b>Entra</b> (ADR-0011)"],
       ["E020-E021", "Análisis de fallos; GT controlada", "0,542 → 0,628 (misma predicción)",
        "Se evalúa con GT controlada (ADR-0012)"],
       ["E022-E024", "Artefacto final; determinismo", "CSV reproducible, sha 252716b0…", "<b>Entregable "
        "actual</b>"],
       ["E023", "Broadcast como fuente de balón", "7,5 % de huecos rellenables", "Descartado"],
       ["E025", "Detector YOLO en el vídeo", "24 % a &lt; 2 m del balón", "Descartado"],
       ["E026", "¿Desfase del broadcast constante?", "Sí, 99 % de los frames", "El audio es utilizable"],
       ["E027", "DFL al banco (GT del tracking)", "Necesita T = 2 s; calibrado sin tocar", "Banco v3"],
       ["E028", "Acierto por longitud de pase", "Pases perdidos: 0,58 tras la pérdida", "Confirma el fallo "
        "nº 1"],
       ["E029", "Silbato en el audio", "Sin señal", "Descartado"],
       ["E030", "Referencia en el banco v3", "<b>0,6470</b>", "Nueva vara de medir"],
       ["E033", "Comentario en inglés (Whisper local)", "Parones fiables, pero ya los detectamos",
        "Fuera del modelo; sirve de comprobación"],
       ["E034", "Modelo profundo (atención + BiGRU) promediado", "0,6570 sobre la referencia", "Pasaba… "],
       ["E035", "<b>Familias de features</b>", "<b>0,7185</b> (+0,072, 9/10)", "<b>Candidato a entrar</b> "
        "(pendiente ablación)"],
       ["E036", "Coste de cambio dinámico (idea PathCRF)", "Sin ganancia", "Descartado"],
       ["E037", "Modelo profundo sobre las familias", "0,7172 / 0,7134", "Descartado: ya no aporta"]],
      [1.8, 5.2, 4.3, 5.7], shade=[(19, GREEN)])
figure(FIG_V3, 16, "Figura 3. Banco v3, partido por partido. Las familias mejoran 9 de 10 partidos; el "
                   "único que baja es J03WR9 (0,661 → 0,631), que estudiaremos.")


# ------------------------------------------------------------------------------------ fallos
H1("7. Dónde fallamos y por qué")
P("El análisis de fallos (E020, <font face='Consolas'>docs/investigacion/02_analisis_fallos.md</font>) "
  "clasifica cada posesión real que no encontramos. Está medido sobre el modelo anterior, en el banco con GT "
  "controlada; las familias nuevas atacan precisamente estos fallos, y repetiremos el análisis con el modelo "
  "nuevo.")
figure(FIG_F, 16, "Figura 4. Causas de las posesiones perdidas. «Equipo contrario» y «fusionada» son dos "
                  "caras del mismo fallo: no ver un cambio de posesión.")
H2("7.1 Robos en juego que no vemos (79 % de las pérdidas)")
bullets([
    "Solo la mitad de los cambios H↔A tenían una frontera nuestra a menos de 2 s. Cuando la detectamos, la "
    "colocamos bien (desfase mediano de 0-1 frame): el problema no es llegar tarde, es no verlo.",
    "Las intercepciones se encontraban el 53 % de las veces; los balones sueltos, el 58 %; tras un pase "
    "perdido acertábamos el 58 % de los frames (E028). En cambio, los cambios de juego y balones largos que "
    "alguien controla se aciertan al 93-95 %.",
    "<b>Por qué</b>: el modelo miraba sobre todo la cercanía al balón y los toques, pero en el ATD los toques "
    "casi no se ven y el balón se pierde justo en el choque. Incluso con el balón perfecto, el jugador más "
    "cercano es del equipo poseedor solo el 79 % de las veces.",
])
H2("7.2 Huecos largos sin balón (38 % de los frames)")
bullets([
    "Acierto por frame: 0,78 en huecos largos frente a 0,92 con balón. El juego vivo se confundía con balón "
    "muerto el 22 % de las veces, y cada D falso parte una posesión.",
    "Rellenar el balón con otras fuentes no funcionó (broadcast 7,5 %, YOLO 24 %): el balón falta en las "
    "mismas jugadas para todos. Hay que inferir sin balón, con la colocación y el movimiento de los jugadores.",
])
H2("7.3 Duelos, aéreos y despejes")
P("Cerca de un duelo acertábamos el 70 % de los frames; las posesiones que acaban en despeje solo se "
  "encontraban el 12,5 %. En 2D no hay altura del balón, y en un despeje la posesión sigue siendo de quien "
  "despeja hasta que el rival controla, pero el modelo cambiaba antes por cercanía.")
H2("7.4 Infrasegmentación")
P("Predecíamos menos posesiones de las que hay (2.298 frente a 2.885 en el banco v3). Es consecuencia del fallo "
  "7.1: cada robo no visto funde posesiones. Con las familias nuevas pasamos a 2.669.")
H2("7.5 Lo que no es problema")
P("El retraso de las fronteras (0-1 frame), los huecos cortos y partir posesiones largas. No merece la pena "
  "invertir ahí.")


# ------------------------------------------------------------------------------------ familias
H1("8. La mejora de esta semana: features como las mira un entrenador")
P("Tras investigar trabajos previos (sección 10) y pensar el problema como un entrenador, reorganizamos las "
  "features en <b>familias</b> (en lugar de versiones v5, v6, v7...). Todo sale de las posiciones, sin "
  "identidades, así que funciona también cuando falta el balón. Cada señal incluye su <b>cambio entre el "
  "segundo anterior y el posterior</b>, porque un robo se ve en el cambio de comportamiento, no en el nivel.")
table([["Familia", "Idea de entrenador", "Cómo se mide"],
       ["Balón", "Hacia dónde avanza; si va en vuelo largo; si está quieto",
        "Velocidad del balón proyectada hacia cada portería, antes y después; balón rápido y lejos de todos "
        "(la posesión se mantiene durante el vuelo); tiempo que lleva quieto"],
       ["Jugadores", "Quién aprieta, quién esprinta, quién se desmarca, qué hace el portero",
        "Jugadores a &lt; 15 m que se acercan al balón a &gt; 3 m/s; nº de sprints (contraataque o "
        "contrapresión); ritmo al que los atacantes se separan de su marcador; el más retrasado como portero "
        "(solo si está a &lt; 25 m de su portería) y si vuelve a su línea"],
       ["Bloques", "Cómo de juntos y abiertos están; si la defensa adelanta o recula; achicar y apretar",
        "Longitud, anchura, superficie (envolvente convexa), dispersión (stretch index), altura de la línea "
        "defensiva y de la delantera, distancia entre líneas y cuánto va adelantado un bloque respecto al "
        "otro. Los niveles, relativos al propio partido"],
       ["Estructura", "Forma con y sin balón; quién va adelantado; cambios de formación",
        "Jugadores por detrás del balón (el que defiende tiene más); jugadores en campo rival; a qué se "
        "parece la forma actual del equipo: a su forma abierta (con balón) o a la cerrada, aprendidas sin "
        "etiquetas en el propio partido; cuánto cambia la forma en 3 s"],
       ["Reanudación", "Un balón parado tiene una colocación típica",
        "Jugadores quietos, área llena (córner, falta), alguien pegado a la banda (saque de banda), alguien "
        "en el área pequeña (saque de puerta), grupo compacto (barrera), formación de saque de centro"]],
      [2.4, 5.0, 9.6])
P("<b>Resultado (E035):</b> 0,7185 frente a 0,6470 (+0,072), mejora en 9 de 10 partidos y las posesiones "
  "predichas suben de 2.298 a 2.669 (reales 2.885). Es la mayor mejora desde el apilado.")
H3("Comprobación de dominio")
P("Antes de evaluar, comparamos la distribución de cada feature nueva en el ATD real y en el banco. Las "
  "medidas de bloque salían desplazadas (en el ATD real los equipos aparecen ~5 m más cortos: puede ser estilo "
  "o jugadores no detectados), así que las expresamos <b>relativas al propio partido</b>, igual que hicimos "
  "con los toques. Los cambios antes/después cuadraban desde el principio.")
box("<b>Cautelas antes de entregarlo.</b> (1) Las features del portero están entre las más importantes y "
    "en el ATD real su distribución difiere algo (desajuste 0,22): la ablación dirá si dependemos de ellas. "
    "(2) Hay que descartar que el modelo aproveche artefactos del simulador (por ejemplo, el balón que el "
    "simulador interpola en los parones). (3) J03WR9 empeora. (4) Falta la prueba de humo en el reto.", RED)


# ------------------------------------------------------------------------------------ descartes
H1("9. Lo que probamos y descartamos (y por qué)")
P("Descartar también es avanzar: cada una de estas vías parecía prometedora y la medimos antes de "
  "invertir más tiempo.")
table([["Idea", "Por qué parecía buena", "Qué pasó", "Conclusión"],
       ["Smart Tagging", "Marca balones parados y el equipo de cada fase", "+0,008 (ruido)",
        "No compensa depender de él"],
       ["Rellenar el balón con el broadcast", "Otra cámara podría ver el balón cuando la táctica no",
        "Solo el 7,5 % de los huecos", "Pierde el balón en las mismas jugadas"],
       ["Detector de balón en el vídeo (YOLO COCO)", "El vídeo táctico es 1080p con homografía",
        "Incluso donde el ATD ve el balón, solo el 24 % a &lt; 2 m", "Balón demasiado pequeño para un "
        "detector genérico. Queda abierta una versión reentrenada con el propio ATD"],
       ["Silbato del árbitro", "Marcaría el inicio de los parones al instante", "Sin señal en el audio",
        "La grada y el comentario lo tapan"],
       ["Comentario en inglés", "Confirmar situaciones y posesiones", "Los parones que comenta ya los "
        "detectamos (techo ~+0,005); el nombre del jugador apenas indica el equipo (0,71 frente a 0,57)",
        "Comprobación independiente, no entra en el modelo"],
       ["Coste de cambio dinámico (PathCRF)", "Abaratar cambiar donde hay señales de robo",
        "La selección nunca lo elige", "La capa 2 ya ve esas señales"],
       ["Modelo profundo (atención + BiGRU)", "Lo social es lo que más aporta en PathCRF; infiere el balón",
        "Sobre la referencia, +0,010; sobre las familias, nada o peor", "Fuera: complica sin aportar"],
       ["Entrenar con la GT controlada", "Optimizar la definición correcta", "−0,002", "Basta con evaluar "
        "con ella"]],
      [3.4, 4.2, 4.4, 5.0])

# ------------------------------------------------------------------------------------ estado del arte
H1("10. No reinventamos la rueda: trabajos que hemos usado")
table([["Trabajo", "Qué aporta", "Cómo lo usamos"],
       ["PathCRF (Kim et al., 2026)", "Posesión sin balón con atención entre jugadores y CRF; 92 % de "
        "acierto de equipo por frame en los mismos 7 partidos DFL", "Inspiró el modelo profundo y el coste "
        "dinámico (ambos medidos y descartados al final) y la idea de que lo colectivo es lo que más aporta"],
       ["Ball Radar / TranSPORTmer", "Inferir la posición del balón a partir de los jugadores",
        "Tarea auxiliar del modelo profundo"],
       ["Pressing Intensity (Bekkers, 2025)", "Presión como probabilidad de interceptar",
        "Familia jugadores: quién aprieta"],
       ["Contrapresión (Bauer y Anzer, 2021)", "Detectar la contrapresión tras pérdida con tracking",
        "Sprints y presión tras el robo"],
       ["Compacidad por fase (AMISCO, LJMU)", "Con balón los equipos son más anchos, largos y extensos",
        "Familia bloques"],
       ["Formaciones (Bialkowski, Shaw y Glickman, SoccerCPD)", "Forma distinta con y sin balón; cambios de "
        "formación", "Familia estructura"],
       ["SoccerNet-Echoes", "Whisper sobre comentarios; falla con los nombres", "Lista de nombres de las "
        "plantillas como pista"]],
      [4.2, 6.0, 6.8])


# ------------------------------------------------------------------------------------ decisiones
H1("11. Decisiones tomadas (ADR) y por qué")
P("Todas están en <font face='Consolas'>docs/decisiones/</font>, en estado <b>Propuesta</b>: falta que las "
  "reviséis y las marquéis como aceptadas.")
table([["ADR", "Decisión", "Motivo"],
       ["0001", "GitHub, ramas y PR con revisión; documentación en el repo", "Los agentes leen el repo, no el "
        "chat; dos personas conocen cada cambio"],
       ["0002", "Los datos del reto nunca van a git", "Licencia y tamaño"],
       ["0003", "Python 3.11 + uv + LightGBM", "Ecosistema de fútbol en Python; entornos reproducibles"],
       ["0004", "Los agentes no ven los datos del reto", "T&amp;C: prohibido enseñar datos o extractos a un "
        "LLM en la nube. <b>Práctica desde el 22 oct (pendiente de que la aprobéis)</b>: el agente ejecuta "
        "scripts de solo agregados con <font face='Consolas'>scripts/run_aggregate.sh</font>, que solo "
        "muestra la salida si acaba bien; si falla, el log lo lee una persona"],
       ["0005-0006", "Validar con un banco externo calibrado con el baseline", "No hay leaderboard"],
       ["0007", "L3 como entrega principal; modelo versionado", "+0,18 sobre el baseline; reproducibilidad"],
       ["0008", "Protocolo de aceptación (≥ 0,01 y mayoría de partidos)", "Evitar quedarnos con la suerte"],
       ["0009", "Balón filtrado en el banco y toques relativos", "El balón real está filtrado"],
       ["0010", "Fusión del balón (broadcast + vídeo)", "Probado: no pasa la puerta"],
       ["0011", "Modelo final: L3 apilado", "+0,029 en 3/3"],
       ["0012", "Evaluar con la posesión controlada", "La GT del banco medía otra definición"],
       ["0013", "Audio del broadcast", "Hay audio; silbato sin señal; comentario solo como comprobación"]],
      [1.8, 6.0, 9.2])
H2("Cumplimiento de las reglas")
bullets([
    "Ningún dato del reto ha pasado por el agente: solo nombres de fichero, tamaños y números agregados "
    "(porcentajes, percentiles, notas). La transcripción del comentario y los logs de error se quedan en "
    "<font face='Consolas'>data/derived/</font>, que el agente no lee.",
    "Los datos no están en git; los derivados (alineación, transcripción) tampoco.",
    "Ninguna etiqueta se ha puesto a mano; la GT privada no se usa; la GT pública solo para humo y para la "
    "forma de la definición de posesión.",
])

# ------------------------------------------------------------------------------------ ahora
H1("12. Qué estamos haciendo ahora y qué viene")
table([["Paso", "Qué", "Estado"],
       ["1", "Ablación: quitar cada familia (empezando por «jugadores», que incluye al portero) para saber "
        "de dónde sale la mejora", "La primera se cortó al reiniciarse la sesión; se relanza"],
       ["2", "Entender por qué empeora J03WR9 y repetir el análisis de fallos con el modelo nuevo", "Pendiente"],
       ["3", "Artefacto final con los 10 partidos y regenerar <font face='Consolas'>La MasIA.csv</font> "
        "(dos veces, mismo sha) + humo con los 5 minutos públicos", "Pendiente; el código ya lo soporta"],
       ["4", "Mejoras baratas: ensamblado de semillas, calibración, reelegir el Viterbi en el banco v3",
        "7-9 nov"],
       ["5", "Congelar el 14 nov: clon limpio que regenere el mismo sha; documento técnico definitivo",
        "10-14 nov"],
       ["6", "Repo público, PR a main aprobada, PDF, subir el formulario", "15-16 nov"]],
      [1.2, 11.0, 4.8])
H2("Qué necesitamos de cada uno")
bullets([
    "<b>Manel</b>: enviar el email a la organización (Q2 agregados y LLM; Q4 versionar derivados; Q5 GT "
    "pública para entrenar; Q6 audio). Invitar a los compañeros en GitHub.",
    "<b>Compañeros</b>: aceptar la invitación, clonar, leer este informe y "
    "<font face='Consolas'>docs/01_contexto_completo.md</font>, revisar y aprobar la PR "
    "<font face='Consolas'>banco-v3 → main</font> y los ADR (sobre todo el 0004).",
    "<b>Los tres</b>: confirmar la inscripción como «La MasIA» antes del 10 nov; uno de nosotros sube el "
    "formulario con su cuenta Google (CSV y PDF con el nombre del equipo).",
])

# ------------------------------------------------------------------------------------ como empezar
H1("13. Cómo empezar y dónde está cada cosa")
P("Instalación (abrir Claude Code <b>desde la carpeta del repo</b>, para que se aplique el bloqueo de "
  "<font face='Consolas'>data/</font>):")
P("git clone --recursive https://github.com/ManelCM/FCBCN_HACK_2027.git C:/dev/hack_barca<br/>"
  "cd C:/dev/hack_barca &amp;&amp; git checkout banco-v3 &amp;&amp; uv sync", code)
P("Datos del reto: del Drive de la organización a <font face='Consolas'>data/</font> y la variable "
  "<font face='Consolas'>MTAH_DATA_DIR</font>. Metrica se descarga solo; DFL, abriendo las URLs en el navegador y "
  "luego <font face='Consolas'>uv run python -m hackbarca.io.dfl --from-downloads &lt;Descargas&gt;</font>.")
table([["Qué", "Dónde"],
       ["Contexto desde cero", "docs/01_contexto_completo.md"],
       ["Plan de investigación y estado del arte", "docs/04_plan_investigacion.md"],
       ["Todos los experimentos con números", "docs/experimentos.md"],
       ["Decisiones", "docs/decisiones/ (ADR 0001-0013)"],
       ["Diario del equipo", "docs/LOG.md"],
       ["Análisis de fallos explicado", "docs/investigacion/02_analisis_fallos.md"],
       ["Features (familias)", "src/hackbarca/features/frame_features.py y tactical.py"],
       ["Modelo y decodificación", "src/hackbarca/models/l3.py, decode/viterbi.py"],
       ["Evaluar en el banco", "uv run python -m hackbarca.pipeline.tune_l3 --bench v3 --features "
        "touch+ball+players+blocks+structure+restart --stack --controlled 1.5 --controlled-dfl 2 --train-gt raw"],
       ["Generar la entrega (datos reales)", "scripts/run_aggregate.sh make_submission uv run python -m "
        "hackbarca.pipeline.make_submission --model l3 --team-name \"La MasIA\""]],
      [5.0, 12.0])

# ------------------------------------------------------------------------------------ glosario
H1("14. Glosario")
table([["Término", "Significado"],
       ["spellF1", "Métrica oficial: F1 de posesiones enteras emparejadas por solape (IoU 0,3-0,7)"],
       ["macroF1", "F1 media por frame de H, A y D; desempate"],
       ["ATD", "Automatic Tracking Data: el tracking automático del reto"],
       ["GT", "Ground truth, la verdad con la que se puntúa"],
       ["GT controlada", "Verdad del banco en la que un robo solo cuenta si dura ≥ T segundos"],
       ["Banco", "Partidos públicos estropeados para parecerse al ATD; nuestro juez"],
       ["Partido fuera (LOGO)", "Entrenar con todos menos uno y puntuar ese"],
       ["Viterbi", "Algoritmo que elige la secuencia de estados más probable cobrando por cada cambio"],
       ["Apilado", "Un segundo modelo que aprende de las probabilidades del primero"],
       ["Ablación", "Quitar una pieza para medir cuánto aporta"],
       ["Oráculo", "Sustituir la predicción por la verdad en ciertos frames para medir el techo de mejora"],
       ["Infrasegmentación", "Predecir menos posesiones de las que hay"]],
      [4.0, 13.0])


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arial", 8)
    canvas.setFillColor(colors.grey)
    if doc.page > 1:
        canvas.drawString(2 * cm, 1.2 * cm, "La MasIA · Informe de estado · 24 oct 2026")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, "Página %d" % doc.page)
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                        bottomMargin=1.8 * cm, title="La MasIA - Informe de estado", author="La MasIA")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("ok", OUT)

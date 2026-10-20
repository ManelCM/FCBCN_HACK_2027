# 00 — El reto: qué sabemos y qué no

_Última actualización: 2026-10-20. Fuentes al final._

## Organización
More Than A Hack, **2ª edición (2027)**. La organizan Mobile World Capital Barcelona (MWCapital) y Barça Innovation Hub (BIHUB), con Metrica Sports como socio de datos. La 1ª edición (feb-mar 2026) tuvo más de 3.200 inscritos de más de 95 países; 12 equipos de 5 llegaron a la final en la Talent Arena del MWC. Esa edición se centraba en **IA generativa aplicada a datos deportivos** y en convertir datos complejos en *insights* tácticos accionables.

## Requisitos de equipo
- **3 a 5 personas**. Con menos de 3 no se puede pasar al Innovation Challenge ni a la final. Somos 3, así que **ninguno puede darse de baja** sin buscar sustituto.
- Inscripción hasta el **10 nov 2026**. **Acción: confirmar que estamos inscritos los 3.**

## Fase 1 — Initial Challenge (20 oct → 17 nov 2026)
- **Tarea**: estimar la **posesión del equipo en cada frame** de un partido.
- **Input**: tracking automático (ATD) de Metrica + **homografía** proporcionada. No hay que hacer detección, identificación ni calibración.
- **Evaluación**: automática, con ranking. Pasan los **20 mejores**.
- Datos disponibles en el reto en general: vídeo de partidos oficiales (BIHUB), tracking automático (Metrica) y eventos automáticos (*Smart Tagging* de Metrica).

## Fase 2 — Innovation Challenge (26 nov → 14 ene 2027)
- Top 20. Dataset de **10 partidos profesionales** (tracking + eventos + vídeo), con mentoría técnica.
- Reto abierto: elegimos preguntas analíticas o tácticas y las resolvemos (análisis, modelado, interpretación, generación de *insights*).
- Pasan **5 equipos** a la final, elegidos por un jurado.

## Fase 3 — Final Pitch (2 mar 2027, presencial)
- Pitch ante un jurado profesional en la Talent Arena (Barcelona). La organización paga viaje y alojamiento.

## ❓ Preguntas abiertas (bloqueantes para la fase 1)

Hay que responderlas **en cuanto tengamos el paquete de datos/las bases**. Quien lo averigüe, que actualice esta tabla y lo anote en el LOG.

| # | Pregunta | Por qué importa | Respuesta |
|---|---|---|---|
| Q1 | ¿Qué **clases** tiene la posesión: {local, visitante} o también {nadie / balón muerto}? | Cambia el modelo y la métrica | ? |
| Q2 | ¿**Métrica** exacta: accuracy por frame, F1 macro, se ignoran los frames de balón muerto...? | Optimizamos lo que se mide | ? |
| Q3 | ¿Cuántos **envíos** al leaderboard se permiten? ¿Es público o privado? ¿Hay partido de *hold-out*? | Riesgo de sobreajuste al leaderboard | ? |
| Q4 | ¿Dan **etiquetas de entrenamiento** (algún partido con la posesión real)? | Supervisado frente a reglas / *weak labels* | ? |
| Q5 | ¿Están disponibles los **eventos Smart Tagging** del partido evaluado? | Si lo están, son la señal más fuerte (ver investigación) | ? |
| Q6 | Formato del tracking: ¿coordenadas de **imagen** (píxeles) o de **campo**? ¿Para qué sirve exactamente la homografía? ¿Es una por frame? | Primer paso del pipeline | ? |
| Q7 | **fps**, IDs de jugador y equipo, ¿hay altura (z) del balón?, ¿huecos cuando el balón no se detecta? | Robustez | ? |
| Q8 | Formato del **fichero de envío** (frame_id, team...) | Para no perder envíos | ? |
| Q9 | ¿Hay reglas sobre uso de modelos externos/LLMs o código abierto? | Cumplimiento | ? |

Contacto de la organización: `hackathon.talentarena@mobileworldcapital.com`. Si algo no queda claro en las bases, **preguntad por correo pronto**.

## Fuentes
- https://landing.mobileworldcapital.com/more-than-a-hack-2027
- https://www.metrica-sports.com/blog/more-than-a-hack-2027-build-something-real-with-fc-barcelona-match-data
- https://www.fcbarcelona.es/es/noticias/4584971/el-bihub-y-mwcapital-unen-fuerzas-para-impulsar-el-talento-digital-con-la-presentacion-de-la-segunda-edicion-de-more-than-a-hack

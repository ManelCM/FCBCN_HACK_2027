# 0022 — Regla del saque inicial: lo anterior al saque de cada parte es balón muerto

- **Estado**: Propuesta (aplicada en la entrega por el agente de Manel el 10 nov; falta la revisión de las otras dos personas)
- **Fecha**: 2026-11-10
- **Autor/a**: Manel (con agente)
- **Revisado por**: ☐ ☐

## Contexto
- FAQ.md y DATA.md oficiales: la 2.ª parte puntúa desde el frame 110.808, pero el saque no llega hasta el
  113.742. Esos 2.934 frames (~2 min) puntúan y **son todos D**. La 1.ª parte puntúa desde su propio saque.
- Punto 5 de ADR-0019 (L0): medir qué predice la entrega en ese tramo y, si hace falta, arreglarlo con una
  **regla algorítmica general**, sin constantes del partido.
- **Medida** (`pipeline/kickoff_report.py`, agregado, entrega E090c con sha `d976e575374b00d5`): en ese
  tramo la entrega da **H 0,450, A 0,086 y D 0,464**. Más de la mitad del tramo sale como juego vivo, así
  que inventamos posesiones antes del saque o alargamos la primera posesión real hasta 2 min hacia atrás.
  Las dos cosas cuestan en spellF1.

## Opciones consideradas
1. **No hacer nada**: el error es pequeño (1-2 posesiones de ~300), pero seguro.
2. **Fijar a D los frames 110.808-113.741** con los números del FAQ. Es exacto, pero es una constante
   ligada a este partido, que ADR-0019 descarta, y se parece demasiado a escribir etiquetas a mano.
3. **Regla general con el Smart Tagging + tracking** (elegida). El Smart Tagging marca cada saque inicial
   como `SET PIECES` con `Type = "Kick Off (Start)"`, en una ventana de ~5 s que contiene el saque (lo
   comprueba el propio `starter/load_smart_tagging.check_time_base`). Dentro de cada parte puntuada:
   - todo lo anterior al inicio de esa ventana es D;
   - dentro de la ventana, el saque es el primer frame en que el balón del tracking se aleja más de 1 m
     de su primera posición vista; si no se ve el balón, se queda en el inicio de la ventana.
4. **Detectar el saque solo con el tracking** (formación de saque de centro + balón que sale del centro):
   no depende del Smart Tagging, pero no se puede validar sin mirar los datos y puede dispararse en un saque
   tras gol.

## Decisión
Opción 3, en `pipeline/kickoff_rule.py`. `make_submission` la aplica si `l3_decode.json` trae
`"kickoff_dead": true`. `train_l3_final` lo escribe por defecto, y `--no-kickoff-dead` la quita.

## Justificación
- Es algorítmica y general: vale para cualquier partido que traiga el Smart Tagging, y usa una entrada
  oficial del reto sin GT.
- No toca el modelo ni el banco. En el banco los partidos empiezan en el saque, así que allí la regla no
  cambia nada y no se puede medir. Se justifica por la definición oficial (FAQ), no por una puntuación.
- El riesgo está acotado: solo actúa antes del primer `Kick Off (Start)` de cada parte. Si el Smart
  Tagging no lo trae, no cambia nada.

## Consecuencias
- Cambia el sha de la entrega. Hay que regenerar `La MasIA.csv` dos veces y repetir `kickoff_report`:
  el tramo anterior al saque debe salir ~100 % D, y el resto del fichero igual que antes.
- El Smart Tagging vuelve a ser una entrada de la entrega, solo para esta regla. Lo dice el documento
  técnico.
- Comprobación de la detección: `kickoff_report --detect` imprime cuántos frames separan el saque
  detectado del publicado en el FAQ. Es una comprobación con la documentación oficial y no ajusta ningún
  parámetro. **Medida el 10 nov**:
  - 1.ª parte, +4 frames (0,16 s tarde). Esa parte puntúa desde el saque, así que solo pasan a D sus
    4 primeros frames;
  - 2.ª parte, −79 frames (3,2 s antes): el balón se mueve más de 1 m dentro de la ventana antes del
    saque. La regla pone a D 2.855 de los 2.934 frames del tramo, y los 3 s restantes los decide el modelo.
  - El umbral de 1 m **no se ajusta** para cuadrar con el FAQ.

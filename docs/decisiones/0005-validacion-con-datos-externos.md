# 0005 — Validar y entrenar con datos externos de tracking etiquetado

- **Estado**: Propuesta (pendiente de verificar qué campos traen DFL y Metrica sample-data)
- **Fecha**: 2026-10-20
- **Autor/a**: Manel
- **Revisado por**: ☐ ☐

## Contexto
No hay leaderboard ni feedback hasta los resultados. La única GT disponible son 5 min (17 posesiones), y el repo oficial demuestra que con eso se escoge mal incluso un solo hiperparámetro. El ruido de spellF1 es ±0,04 con un partido completo, así que con 17 posesiones es mucho mayor.

## Opciones consideradas
1. **Ajustar con los 5 min públicos**: es exactamente el error que el repo pide evitar.
2. **Etiquetar a mano minutos del partido para validar**: ambiguo según las reglas (Q1), caro, y una persona no reproduce la definición mecánica de Metrica.
3. **Datasets abiertos con tracking corregido que traigan balón vivo/muerto y posesión por frame** (DFL/Sportec open data, Metrica sample-data, quizá SkillCorner o PFF), degradados para que se parezcan al ATD → validación con cientos o miles de posesiones.
4. **Juicio "a ojo" + reglas razonables**: no se puede medir.

## Decisión (propuesta)
Opción 3 como validación principal y fuente de entrenamiento. Los 5 min públicos se usan solo como prueba de humo y para medir el sesgo de las fronteras.

## Justificación
- RULES.md permite explícitamente datos externos.
- Las etiquetas de esos datasets se generan como la GT del reto (flags del tracking corregido), así que optimizamos el mismo concepto.
- Muestra grande → diferencias entre modelos estadísticamente significativas (bootstrap con el scorer oficial).
- También es un argumento potente para el documento técnico.

## Consecuencias
- Primera tarea: **verificar** campos y licencias de cada dataset (tarea de 1 día).
- Hay que construir el **simulador de degradación** (balón ausente, tracks fragmentados) calibrado con estadísticas agregadas de nuestro ATD.
- Riesgo de *domain shift* (otra liga, otra cámara, otra definición de proveedor): se mitiga comprobando que el ranking de modelos en datos externos coincide con el de los 5 min públicos en las diferencias grandes.

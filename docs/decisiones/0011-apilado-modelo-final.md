# 0011 — Modelo final: L3 apilado (dos capas) con features v3, sin aleatorización del dominio

- **Estado**: Propuesta
- **Fecha**: 2026-10-22
- **Autor/a**: Manel (con agente)
- **Revisado por**: ☐ ☐

## Contexto
El análisis de errores (E009) muestra que lo que más spellF1 cuesta es la segmentación de equipo,
y que la capa única de LightGBM ve cada frame casi aislado: solo con ventanas de features de
hasta ±5 s. Faltaba contexto temporal largo sobre las propias predicciones.

## Opciones consideradas
1. **L3 v3 (una capa)**: 0,513 en el banco v2.1 (float32).
2. **Apilado**: una 2.ª capa LightGBM recibe las features más las probabilidades de la 1.ª capa
   resumidas hacia atrás, centradas y hacia delante en 1/2/5/10 s. La 1.ª capa da probabilidades
   fuera de muestra (validación interna) para que la 2.ª no aprenda de predicciones demasiado
   buenas.
3. **Apilado + aleatorización del dominio** (3 degradaciones extra por partido).
4. Modelo secuencial con torch (GRU/TCN): más complejo, con dependencia de GPU para entrenar y
   más difícil de hacer determinista.

## Decisión
**Opción 2**: L3 apilado, features v3, sin Smart Tagging y sin aleatorización.
Artefacto: `l3_lgbm.txt.gz` (capa 1) + `l3_stage2.txt.gz` (capa 2) + `l3_decode.json` con
`"stack": true`. Se genera con
`train_l3_final --features 3 --no-tags --stack --temperature 1 --switch 20 --to-dead 10 --from-dead 10`
(la decodificación que mejor sale en los 3 partidos, E018).

## Justificación
Banco v2.1, dejando un partido fuera (ADR-0008):

| modelo | game 1 | game 2 | game 3 | total |
|---|---|---|---|---|
| v3, una capa | 0,530 | 0,497 | 0,513 | 0,513 |
| **v3 apilado** | **0,567** | **0,522** | **0,537** | **0,542** |
| v3 apilado + aleatorización | 0,555 | 0,545 | 0,549 | 0,550 |

- El apilado pasa el protocolo: +0,029 y mejora en los 3 partidos.
- La aleatorización **no** lo pasa sobre el apilado (+0,008, peor en el game 1).
- **Robustez** (partido retenido con la calibración alternativa del balón): una capa 0,518 →
  **apilado 0,558** (0,568/0,534/0,572). El apilado ya es robusto sin aleatorizar. La prueba de
  apilado + aleatorización con el dominio desplazado se cortó por falta de memoria; si al
  repetirla mejora claramente, se revisa esta decisión.

## Consecuencias
- La predicción cuesta el doble (dos modelos), sigue siendo solo CPU y cabe en segundos.
- El artefacto ocupa ~2 × 7,6 MB en git.
- `make_submission --model l3` detecta el apilado por `l3_decode.json`; no cambia el comando.

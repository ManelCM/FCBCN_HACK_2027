# 0024 — Criterio de entrega: manda el banco, con los límites absolutos de D en el reto como veto

- **Estado**: Aceptada por Manel el 12 nov (falta la revisión de las otras dos personas)
- **Fecha**: 2026-11-12
- **Autor/a**: Manel (con agente)
- **Revisado por**: ☐ ☐
- **Sustituye**: el veto relativo de ADR-0020. Mantiene ADR-0008 (protocolo del banco) y ADR-0023 (sin GT ni vídeo del reto).

## Contexto
ADR-0020 deja entrar un candidato solo si no empeora **ninguna** de las cuatro medidas de forma del reto frente
a la entrega vigente. Desde el 10 nov, la vigente es E110 (E058b sin vídeo, 0,7355 en el banco con la regla de
3 s). Los candidatos posteriores mejoran mucho el banco y caen en ese veto, casi siempre por los cambios directos H↔A:

| candidato | banco (LOGO, regla de 3 s) | D < 5 s/min | tramos D/min | cambios directos/min | % de D |
|---|---|---|---|---|---|
| E110 (vigente) | 0,7355 | 0,274 | 1,360 | 1,539 | 0,436 |
| E117, decodificación de la LOGO | 0,7513 | 0,217 | 1,218 | 1,416 | 0,434 |
| E117b, 1/30/20/20 | 0,7438 | 0,283 | 1,350 | 1,652 | 0,431 |
| **E122**, E113 + poseedor al 30 %, decodificación de la LOGO | **0,7621** | **0,142** | **1,039** | 1,435 | 0,442 |
| E122b, 1/30/20/20 | 0,7610 | 0,264 | 1,228 | 1,492 | 0,439 |

Límites absolutos de ADR-0018: D < 5 s ≤ 0,25; tramos D entre 0,8 y 1,3; cambios directos ≥ 1,7; % de D
entre 0,38 y 0,50. Ninguna entrega sin vídeo ha llegado nunca a 1,7 cambios directos. En el banco, la GT tiene
2,03.

## Opciones consideradas
1. **Mantener ADR-0020 tal cual**: la entrega se queda en E110 y se pierde todo lo que mejora el banco desde el
   10 nov (hasta +0,027 con la regla).
2. **ADR-0020 con una tolerancia** (p. ej., un 5 % por medida): entraría E117b, pero no E122.
3. **Manda el banco, con un veto absoluto** en las medidas de D, que son las que miden el defecto conocido del
   reto (D cortos que parten posesiones, ADR-0018/0021). Los cambios directos se informan, pero no vetan.

## Decisión
**Opción 3.** Un candidato sustituye a la entrega si cumple todo esto:
1. Pasa ADR-0008 en el banco con la configuración de la entrega, es decir, con la regla de D cortos de 3 s.
2. En el reto (agregados de nuestra predicción, `real_vs_bench`) cumple los límites absolutos de ADR-0018 en:
   - D < 5 s: ≤ 0,25 por minuto;
   - tramos D: entre 0,8 y 1,3 por minuto;
   - % de D: entre 0,38 y 0,50.
3. `make_submission` da el mismo sha en dos ejecuciones.

Entre los candidatos que lo cumplen, **gana el mejor en el banco**. El orden se fija antes de mirar el reto.
Las cuatro medidas solo vetan: nunca se elige por la forma en el reto (ADR-0023).

Aplicado el 12 nov: **E122 con la decodificación de la LOGO** pasa a ser la entrega. Es la primera del orden
pre-registrado y cumple los tres límites. E122b no entra (0,264 D cortos).

## Justificación
- El banco es la única medida de acierto que tenemos. Las medidas del reto describen la forma de nuestra
  predicción y no dicen si acierta.
- Los D cortos y los tramos D miden el defecto conocido del reto. E090b dice que esos D cortos son errores en
  su mayoría. E122 es el primer candidato sin vídeo que los deja dentro de los límites de ADR-0018.
- Los cambios directos bajan en todos los candidatos que mejoran el banco. No sabemos si en el reto eso es un
  error, porque no hay GT. El riesgo se acepta de forma explícita y se declara en el documento técnico.

## Consecuencias
- La entrega usa el poseedor (`models/artifacts/possessor.pt`), entrenado en GPU con los 10 partidos del banco.
  - La predicción va en CPU y es determinista.
  - Reentrenar en GPU no garantiza los mismos bits, así que los pesos se versionan.
- Riesgo declarado: en el reto, la predicción cambia de equipo directamente menos que la GT del banco
  (1,44 frente a 2,03 por minuto), sobre todo en los huecos de balón.
- Se revisa si llega un candidato nuevo (E115/E114) o una respuesta de la organización.

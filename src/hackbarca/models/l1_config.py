"""Parametros congelados de L1 para la entrega. Elegidos en el banco externo (E004), nunca con datos
del reto. Cambiarlos exige un experimento nuevo en docs/experimentos.md."""

from hackbarca.models.l1 import L1Params

# E004: trial 97 de `tune_l1 --trials 100 --seed 0`, mejor media en los 3 partidos del banco.
L1_FROZEN = L1Params(
    max_gap=100,
    dist_scale=1.1291943830731728,
    p_dead_visible=0.08111725763493209,
    p_dead_gap=0.7590529536848931,
    p_dead_out=0.97,
    out_margin=1.0,
    switch=98.74590907749969,
    to_dead=84.86696268281898,
    from_dead=6.705271656343322,
    tag_before=85,
    p_tag_before=0.855978254170374,
    p_tag_in=0.28194453942473735,
)

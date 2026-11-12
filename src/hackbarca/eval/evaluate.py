"""Wrapper fino sobre evaluation/possession_metrics.py del repo oficial.

No reimplementa la metrica (seria divergir de la especificacion oficial), solo
expone una API de libreria comoda para nuestro pipeline: validar un CSV de
envio y evaluarlo contra una GT, con bootstrap, devolviendo un resultado
agregado facil de registrar en docs/experimentos.md.

ADR-0004: esto NUNCA debe ejecutarse con datos reales del reto desde un agente.
Las unicas GT seguras para un agente son la publica de 5 min (es publica, esta
en el repo oficial) y GT sinteticas de test. La GT privada del partido completo
la ejecuta una persona.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
_EVAL_DIR = os.path.join(_REPO_ROOT, "external", "more-than-a-hack-2027", "evaluation")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

try:
    import possession_metrics as pm  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "no se encuentra possession_metrics.py del repo oficial en {}. "
        "Inicializa el submodulo: git submodule update --init".format(_EVAL_DIR)
    ) from exc


@dataclass
class ValidationResult:
    ok: bool
    reason: str | None = None
    n_rows: int | None = None
    label_counts: dict = field(default_factory=dict)
    n_spells: int | None = None
    mean_spell_duration_s: float | None = None

    def __bool__(self) -> bool:
        return self.ok


def validate_csv(path: str, expected_rows: int = 199319, fps: int = 25) -> ValidationResult:
    """Valida un CSV frame,label con la misma logica que evaluation/validate_submission.py.

    No necesita ground truth. Solo comprueba la forma del fichero (199319 filas,
    frame 0..199318, header frame,label, alfabeto H/A/D/X) y describe lo que
    el fichero afirma (recuento de labels, nº de posesiones que declara).
    """
    try:
        table = pm.read_label_csv(path)
    except Exception as error:
        return ValidationResult(ok=False, reason="no se pudo leer: {}".format(error))

    reason = pm.validate_submission(table, expected_rows)
    if reason:
        return ValidationResult(ok=False, reason=reason, n_rows=len(table))

    codes = pm.label_codes(table)
    counts = {name: int((codes == index).sum()) for index, name in enumerate(pm.LABELS)}

    runs = pm.valid_runs(codes)
    spells = pm.possession_spells(codes, runs)
    import numpy as np
    n_alive = int(np.sum((codes == pm.H) | (codes == pm.A)))
    mean_duration = (n_alive / float(fps) / len(spells)) if spells else 0.0

    return ValidationResult(
        ok=True, n_rows=len(codes), label_counts=counts,
        n_spells=len(spells), mean_spell_duration_s=mean_duration,
    )


def load_gt_codes(gt_path: str):
    """Codigos 0..3 de un CSV de ground truth (publica o sintetica)."""
    return pm.label_codes(pm.read_label_csv(gt_path))


def evaluate_submission(submission_path: str, gt_path: str, taus=None, bootstrap_n: int = 200):
    """Evalua un envio contra una GT. Devuelve (result_dict, reason).

    `result_dict` es exactamente lo que devuelve `possession_metrics.evaluate`
    (macro_f1, spell_f1_mean, intervalos de bootstrap, transiciones, etc.), o
    None si el fichero no es un envio valido -- en ese caso `reason` lo explica.
    Replica `run_evaluation.score_submission` pero como funcion de libreria,
    sin pasar por el bucle de ficheros en results/.
    """
    taus = tuple(sorted(set(taus))) if taus else pm.TAUS
    try:
        table = pm.read_label_csv(submission_path)
    except Exception as error:
        return None, "no se pudo leer: {}".format(error)

    gt_codes = load_gt_codes(gt_path)
    reason = pm.validate_submission(table, len(gt_codes))
    if reason:
        return None, reason

    result = pm.evaluate(gt_codes, pm.label_codes(table), taus=taus, bootstrap_n=bootstrap_n)
    return result, None


def summarize(result: dict) -> str:
    """Una linea de resumen apta para pegar en docs/experimentos.md."""
    return (
        "macroF1={macro_f1:.4f}  spellF1={spell_f1_mean:.4f}  "
        "poss={n_pred_spells} (gt {n_gt_spells})  dur={mean_spell_duration_s:.1f}s "
        "(gt {gt_mean_spell_duration_s:.1f}s)".format(**result)
    )

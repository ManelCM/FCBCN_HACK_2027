"""Un unico comando para regenerar submissions/<equipo>.csv con el baseline oficial.

Reutiliza starter/baseline.py del repo oficial (nearest-to-ball + voto mayoritario
en ventana) sin reimplementarlo, y valida el CSV resultante con nuestro wrapper de
evaluacion antes de darlo por bueno. Esta es la submission de RESERVA: siempre debe
haber una valida subida (ver docs/02_estrategia.md), aunque luego la superemos con
L1/L2/L3.

ADR-0004: este script LEE data/ (tracking real). Un agente en la nube NO debe
ejecutarlo -- que lo corra una persona. El comando exacto es:

    uv run python -m hackbarca.pipeline.make_baseline_submission --team-name <equipo>

Por defecto usa window=50 (el "suelo publicado" en EVALUATION.md, spellF1=0.2433
en partido completo -- NUNCA elegir la ventana por el spellF1 de los 5 min
publicos, ver docs/investigacion/01_posesion_estado_del_arte.md).
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_OFFICIAL_REPO = os.path.join(_REPO_ROOT, "external", "more-than-a-hack-2027")
_STARTER_DIR = os.path.join(_OFFICIAL_REPO, "starter")
for _path in (_OFFICIAL_REPO, _STARTER_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

DEFAULT_WINDOW = 50  # el "suelo publicado": spellF1=0.2433 en partido completo (EVALUATION.md)
DEFAULT_SUBMISSIONS_DIR = os.path.join(_REPO_ROOT, "submissions")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team-name", required=True,
                        help="nombre del equipo tal como se declaro en el formulario "
                             "de inscripcion -- el fichero se llama <team-name>.csv")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help="semiancho de la ventana de voto mayoritario, en frames "
                             "(default: {}, el suelo publicado)".format(DEFAULT_WINDOW))
    parser.add_argument("--out-dir", default=DEFAULT_SUBMISSIONS_DIR,
                        help="directorio de salida (default: submissions/)")
    parser.add_argument("--data-dir", default=None,
                        help="override de MTAH_DATA_DIR (default: variable de entorno)")
    args = parser.parse_args(argv)

    import baseline  # starter/baseline.py del repo oficial
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
    from hackbarca.eval.evaluate import validate_csv  # noqa: E402

    print("Leyendo el ATD y calculando el jugador mas cercano al balon por frame...")
    codes = baseline.build(data_dir=args.data_dir)

    labels = baseline.smooth(codes, args.window)

    out_path = os.path.join(args.out_dir, "{}.csv".format(args.team_name))
    baseline.write_submission(out_path, labels)
    print("Escrito {}".format(out_path))

    result = validate_csv(out_path)
    if not result.ok:
        print("VALIDACION FALLIDA: {}".format(result.reason))
        return 1

    print("Validacion OK: {} filas, {} posesiones declaradas de duracion media {:.1f}s".format(
        result.n_rows, result.n_spells, result.mean_spell_duration_s))
    print("Recuento de labels: {}".format(result.label_counts))
    print("\nSiguiente paso: puntuarlo con el scorer oficial (lee solo CSVs, no necesita data/):")
    print("    cd external/more-than-a-hack-2027 && python evaluation/run_evaluation.py "
          "--results ../../submissions --gt private")
    return 0


if __name__ == "__main__":
    sys.exit(main())

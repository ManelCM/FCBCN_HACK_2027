"""L0 (ADR-0019): que predice la entrega antes del saque de la 2.a parte. Solo agregados.

FAQ.md oficial: la 2.a parte puntua desde el frame 110.808 pero el saque llega en el 113.742; esos
2.934 frames puntuan y son todos D. No fijamos ninguna constante del partido en el modelo: aqui solo se
mide, con el numero publicado en el FAQ, que parte de ese tramo sale D en nuestro fichero.

    scripts/run_aggregate.sh kickoff uv run python -m hackbarca.pipeline.kickoff_report
"""

from __future__ import annotations

import argparse
import hashlib
import os

import pandas as pd

from hackbarca.pipeline.atd_stats import GRADED

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
KICKOFF_2ND = 113742   # FAQ.md
KICKOFFS_FAQ = (11185, 113742)   # FAQ.md y DATA.md: la 1.a parte puntua desde su saque


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(_REPO_ROOT, "submissions", "La MasIA.csv"))
    parser.add_argument("--detect", action="store_true",
                        help="ADR-0022: frames entre el saque que detecta la regla y el publicado en el FAQ")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)
    if args.detect:
        from hackbarca.io.load_match import load_match
        from hackbarca.pipeline.kickoff_rule import kickoff_frames, load_events
        kicks = kickoff_frames(load_events(args.data_dir), load_match(data_dir=args.data_dir), list(GRADED))
        for half, (kick, faq) in enumerate(zip(kicks, KICKOFFS_FAQ), 1):
            print("parte %d: saque detectado - FAQ = %s frames" % (half, "sin tag" if kick is None else kick - faq))
    with open(args.csv, "rb") as handle:
        sha = hashlib.sha256(handle.read()).hexdigest()[:16]
    labels = pd.read_csv(args.csv)["label"].str.strip().str.upper().to_numpy()
    start = GRADED[1][0]
    part = labels[start:KICKOFF_2ND]
    print("fichero sha256 %s" % sha)
    print("tramo antes del saque de la 2.a parte: %d frames" % len(part))
    for code in "HADX":
        print("  %s: %.3f" % (code, (part == code).mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

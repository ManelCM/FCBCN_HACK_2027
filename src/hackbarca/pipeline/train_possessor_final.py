"""E122: entrena el poseedor (E078) con los 10 partidos del banco v3.1 y guarda sus pesos para la entrega.

La entrega lo mezcla en logit con el L3 (E121, `logit_blend.blend`) si `l3_decode.json` lleva
`possessor_weight` > 0. Se entrena en GPU si la hay (semilla fija; la GPU no garantiza los mismos bits al
reentrenar, por eso los pesos se versionan) y se predice siempre en CPU con algoritmos deterministas.

    uv run python -m hackbarca.pipeline.train_possessor_final [--epochs 6]
"""

from __future__ import annotations

import argparse
import os

from hackbarca.pipeline.bench_config import BENCH_GAMES_V3
from hackbarca.pipeline.train_l3_final import ARTIFACTS

POSSESSOR_PATH = os.path.join(ARTIFACTS, "possessor.pt")


def load_possessor(path: str = POSSESSOR_PATH):
    import torch

    from hackbarca.models.possessor import build_model
    model = build_model()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6, help="las mismas que el OOF de E078")
    args = parser.parse_args(argv)
    import torch

    from hackbarca.models.possessor import train_possessor
    from hackbarca.pipeline.tune_possessor import game_inputs
    data = [game_inputs(g, i) for i, g in enumerate(BENCH_GAMES_V3)]
    model = train_possessor([d[:5] for d in data], epochs=args.epochs)
    os.makedirs(ARTIFACTS, exist_ok=True)
    torch.save(model.state_dict(), POSSESSOR_PATH)
    print("guardado", POSSESSOR_PATH, "%.1f MB" % (os.path.getsize(POSSESSOR_PATH) / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

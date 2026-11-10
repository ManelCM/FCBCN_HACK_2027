"""Reproducibilidad del YOLO afinado (variante A): compara, tensor a tensor, dos entrenamientos.

    uv run --group vision python -m hackbarca.vision.pseudo_labels train --name atd_pl_repro
    scripts/run_aggregate.sh weights uv run --group vision python -m hackbarca.vision.weights_check

Solo imprime si los pesos son identicos y la diferencia maxima.
"""

from __future__ import annotations

import argparse
import os

from hackbarca.vision.pseudo_labels import out_dir


def _state(path: str) -> dict:
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ckpt.get("ema") or ckpt["model"]
    return {k: v.float() for k, v in model.state_dict().items()}


def main(argv=None) -> int:
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="atd_pl")
    parser.add_argument("--b", default="atd_pl_repro")
    args = parser.parse_args(argv)
    runs = os.path.join(out_dir(), "runs")
    a = _state(os.path.join(runs, args.a, "weights", "best.pt"))
    b = _state(os.path.join(runs, args.b, "weights", "best.pt"))
    same_keys = a.keys() == b.keys()
    diffs = [float((a[k] - b[k]).abs().max()) for k in a if k in b and a[k].shape == b[k].shape]
    equal = same_keys and all(torch.equal(a[k], b[k]) for k in a)
    print("tensores %d / %d; identicos: %s; diferencia maxima %.3g" % (len(a), len(b), equal, max(diffs or [0.0])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

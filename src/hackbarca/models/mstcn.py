"""MS-TCN (Farha y Gall 2019) sobre las features por frame (M5c del plan completo, E079).

Segmentacion temporal por etapas: cada etapa son capas de convolucion 1D dilatada (residuales) que
refinan la salida de la anterior, con la perdida de suavizado del articulo (MSE truncado sobre las
log-probabilidades de frames vecinos) contra la sobresegmentacion. Entrada: las features de la cache
de tune_l3 (tipificadas, NaN -> 0 mas una mascara por columna) a 5 fps. Salida: P(H/A/D) a 25 fps,
simetrizada con los equipos intercambiados (columnas _h/_a y signos, como swap_teams).
"""

from __future__ import annotations

import numpy as np

SEED = 0
STEP = 5
TAU = 4.0          # truncado de la perdida de suavizado (articulo)
LAMBDA = 0.15


def build_model(n_in: int, ch: int = 64, stages: int = 3, layers: int = 9):
    import torch
    from torch import nn

    class Stage(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.inp = nn.Conv1d(n_in, ch, 1)
            self.layers = nn.ModuleList([nn.Conv1d(ch, ch, 3, padding=2 ** i, dilation=2 ** i) for i in range(layers)])
            self.mix = nn.ModuleList([nn.Conv1d(ch, ch, 1) for _ in range(layers)])
            self.out = nn.Conv1d(ch, 3, 1)
            self.drop = nn.Dropout(0.2)

        def forward(self, x):
            h = self.inp(x)
            for conv, mix in zip(self.layers, self.mix):
                h = h + self.drop(mix(torch.relu(conv(h))))
            return self.out(h)

    class MSTCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = Stage(n_in)
            self.rest = nn.ModuleList([Stage(3) for _ in range(stages - 1)])

        def forward(self, x):
            outs = [self.first(x)]
            for st in self.rest:
                outs.append(st(torch.softmax(outs[-1], 1)))
            return outs

    return MSTCN()


def prepare(df, mu=None, sd=None):
    """(x (F, T) a 5 fps, mu, sd): tipificado, NaN -> 0 y fraccion de NaN como canal extra."""
    v = df.to_numpy(np.float32)[::STEP]
    if mu is None:
        mu, sd = np.nanmean(v, 0), np.nanstd(v, 0) + 1e-6
    z = np.clip((v - mu) / sd, -6, 6)
    nan = np.isnan(z)
    z[nan] = 0
    return np.concatenate([z, nan.mean(1, keepdims=True)], 1).T.astype(np.float32), mu, sd


def train_mstcn(xs, ys, epochs: int = 40, device=None, chunk: int = 2000):
    """xs: [(F, T)], ys: [(T,) 0-3]. Entrena por trozos de `chunk` pasos (400 s)."""
    import torch
    from torch import nn
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(xs[0].shape[0]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    ce = nn.CrossEntropyLoss(ignore_index=3)
    pieces = [(x[:, s:s + chunk], y[s:s + chunk]) for x, y in zip(xs, ys) for s in range(0, x.shape[1], chunk // 2)]
    for _ in range(epochs):
        for i in rng.permutation(len(pieces)):
            x, y = pieces[i]
            if x.shape[1] < 50:
                continue
            xt = torch.from_numpy(x[None]).to(device)
            yt = torch.from_numpy(y.astype(np.int64)).to(device)
            loss = 0
            for out in model(xt):
                loss = loss + ce(out[0].T, yt)
                lp = torch.log_softmax(out, 1)
                diff = (lp[:, :, 1:] - lp[:, :, :-1].detach()) ** 2
                loss = loss + LAMBDA * torch.clamp(diff, max=TAU ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.to("cpu").eval()


def predict_mstcn(model, x, x_swapped, n_full: int) -> np.ndarray:
    import torch
    with torch.no_grad():
        p = torch.softmax(model(torch.from_numpy(x[None]))[-1], 1)[0].numpy().T
        q = torch.softmax(model(torch.from_numpy(x_swapped[None]))[-1], 1)[0].numpy().T
    p5 = (p + q[:, [1, 0, 2]]) / 2
    full = np.repeat(p5, STEP, axis=0)[:n_full]
    if len(full) < n_full:
        full = np.vstack([full, np.repeat(full[-1:], n_full - len(full), axis=0)])
    return full / full.sum(1, keepdims=True)

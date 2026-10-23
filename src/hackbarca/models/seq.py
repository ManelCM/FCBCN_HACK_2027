"""Modelo profundo de posesion (plan §4.3, E034), inspirado en PathCRF y Ball Radar (Kim et al.).

Por frame (a 5 fps) el conjunto de jugadores visibles es una "frase" sin orden: atencion entre
jugadores (no necesita identidades estables, encaja con los slots fragmentados del ATD), resumen por
equipo, y una BiGRU sobre ventanas de 10 s. Dos salidas:
- P(H/A/D) por frame;
- tarea auxiliar: la posicion del balon (aprendida donde el banco limpio la tiene), para que el
  modelo aprenda donde esta el balon dentro de los huecos.

Se usa como segunda fuente de probabilidades, promediada con LightGBM antes del Viterbi. Entrena en
GPU si hay; predice en CPU con algoritmos deterministas (la entrega no necesita GPU).
"""

from __future__ import annotations

import numpy as np

STEP = 5          # 25 fps -> 5 fps
WINDOW = 50       # 10 s a 5 fps
MAX_TOKENS = 26   # jugadores visibles por frame (el resto se descarta)
N_TOKEN_FEATS = 8
N_GLOBAL_FEATS = 4
SEED = 0


def frame_tokens(match, ball_x=None, ball_y=None):
    """(tokens (n, MAX_TOKENS, F), mascara (n, MAX_TOKENS), global (n, G)) a 25 fps.

    Token: x, y normalizados, velocidad (vx, vy) en 1 s, equipo (+1 local / -1 visitante),
    distancia al balon y su indicador, y si el jugador es el mas cercano al balon de su equipo.
    Global: balon x, y, visible y longitud del hueco (log).
    """
    from hackbarca.models.l1 import _gap_lengths
    n, length, width = match.n_frames, match.pitch_length, match.pitch_width
    px, py = match.players_x, match.players_y
    bx = match.ball_x if ball_x is None else ball_x
    by = match.ball_y if ball_y is None else ball_y
    h = match.fps // 2
    vx = np.full(px.shape, np.nan)
    vy = np.full(py.shape, np.nan)
    vx[h:-h] = (px[2 * h:] - px[:-2 * h]) * match.fps / (2 * h)
    vy[h:-h] = (py[2 * h:] - py[:-2 * h]) * match.fps / (2 * h)
    team = np.where(match.team_of_track == 0, 1.0, -1.0)
    known = ~np.isnan(bx)
    d = np.hypot(px - np.nan_to_num(bx)[:, None], py - np.nan_to_num(by)[:, None])
    order = np.argsort(np.where(np.isnan(px), np.inf, np.where(known[:, None], d, 0.0)), axis=1)[:, :MAX_TOKENS]
    rows = np.arange(n)[:, None]
    sel_x, sel_y = px[rows, order], py[rows, order]
    mask = ~np.isnan(sel_x)
    dd = np.where(known[:, None], d[rows, order], np.nan)
    closest = np.zeros_like(sel_x)
    for t_val in (1.0, -1.0):
        dteam = np.where(team[order] == t_val, np.nan_to_num(dd, nan=np.inf), np.inf)
        idx = np.argmin(dteam, axis=1)
        ok = np.isfinite(dteam[np.arange(n), idx])
        closest[np.arange(n)[ok], idx[ok]] = 1.0
    tokens = np.stack([
        sel_x / length - 0.5, sel_y / width - 0.5,
        np.clip(vx[rows, order], -12, 12) / 8, np.clip(vy[rows, order], -12, 12) / 8,
        np.broadcast_to(team[order], sel_x.shape),
        np.nan_to_num(np.clip(dd, 0, 60) / 30, nan=0.0), np.isnan(dd).astype(float), closest,
    ], axis=-1)
    tokens = np.where(mask[..., None], np.nan_to_num(tokens), 0.0).astype(np.float32)
    gap = np.log1p(_gap_lengths(~known))
    glob = np.stack([np.nan_to_num(bx / length - 0.5), np.nan_to_num(by / width - 0.5),
                     known.astype(float), gap / 7], axis=-1).astype(np.float32)
    return tokens, mask, glob


def build_model():
    import torch
    from torch import nn

    class SetGRU(nn.Module):
        def __init__(self, d: int = 64, heads: int = 4):
            super().__init__()
            self.embed = nn.Sequential(nn.Linear(N_TOKEN_FEATS, d), nn.ReLU(), nn.Linear(d, d))
            layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.1, batch_first=True)
            self.attn = nn.TransformerEncoder(layer, num_layers=2)
            self.glob = nn.Linear(N_GLOBAL_FEATS, d)
            self.gru = nn.GRU(4 * d, d, batch_first=True, bidirectional=True)
            self.cls = nn.Linear(2 * d, 3)
            self.ball = nn.Linear(2 * d, 2)

        def forward(self, tokens, mask, glob):
            b, t, k, f = tokens.shape
            x = self.embed(tokens.reshape(b * t, k, f))
            m = mask.reshape(b * t, k)
            empty = ~m.any(1)
            m = m.clone()
            m[empty, 0] = True   # un frame sin jugadores no puede quedar sin ningun token
            x = self.attn(x, src_key_padding_mask=~m)
            team = tokens.reshape(b * t, k, f)[..., 4]
            pooled = []
            for sign in (1.0, -1.0):   # resumen por equipo (media de sus tokens)
                w = (m & (team == sign)).float().unsqueeze(-1)
                pooled.append((x * w).sum(1) / w.sum(1).clamp(min=1.0))
            allm = m.float().unsqueeze(-1)
            pooled.append((x * allm).sum(1) / allm.sum(1).clamp(min=1.0))
            pooled.append(self.glob(glob.reshape(b * t, -1)))
            z, _ = self.gru(torch.cat(pooled, -1).reshape(b, t, -1))
            return self.cls(z), self.ball(z)

    return SetGRU()


def _windows(n: int):
    starts = list(range(0, max(n - WINDOW, 0) + 1, WINDOW // 2))
    if starts[-1] + WINDOW < n:
        starts.append(n - WINDOW)
    return starts


def _swap(tokens, glob, y):
    """Mismo partido con los equipos intercambiados (aumento de datos y prediccion simetrica)."""
    t = tokens.copy()
    t[..., 4] *= -1
    yy = np.where(y == 0, 1, np.where(y == 1, 0, y))
    return t, glob, yy


def train_seq(games, epochs: int = 6, device: str | None = None):
    """games: [(tokens, mask, glob, labels 0/1/2/3, ball_xy limpio (n,2) con NaN)] ya a 5 fps."""
    import torch
    from torch import nn
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    items = []
    for tokens, mask, glob, y, ball in games:
        for s in _windows(len(y)):
            items.append((tokens[s:s + WINDOW], mask[s:s + WINDOW], glob[s:s + WINDOW], y[s:s + WINDOW], ball[s:s + WINDOW]))
    ce = nn.CrossEntropyLoss(ignore_index=3)
    for _ in range(epochs):
        order = rng.permutation(len(items))
        for a in range(0, len(order), 64):
            batch = [items[i] for i in order[a:a + 64]]
            tk, mk, gl, yy, bl = (np.stack(c) for c in zip(*batch))
            if rng.random() < 0.5:
                tk, gl, yy = _swap(tk, gl, yy)
            tk, mk, gl = (torch.from_numpy(v).to(device) for v in (tk, mk, gl))
            logits, ball_hat = model(tk, mk, gl)
            loss = ce(logits.reshape(-1, 3), torch.from_numpy(yy.astype(np.int64)).to(device).reshape(-1))
            target = torch.from_numpy(np.nan_to_num(bl)).to(device)
            has = torch.from_numpy(~np.isnan(bl).any(-1)).to(device)
            if has.any():
                loss = loss + 0.5 * ((ball_hat - target) ** 2).sum(-1)[has].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.to("cpu").eval()


def predict_seq(model, tokens, mask, glob, n_full: int) -> np.ndarray:
    """P(H/A/D) a 25 fps (n_full frames) desde entradas a 5 fps, simetrizada; CPU determinista."""
    import torch
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)   # solo durante la prediccion (en CPU); se restaura al final
    try:
        return _predict(model, tokens, mask, glob, n_full)
    finally:
        torch.use_deterministic_algorithms(previous)


def _predict(model, tokens, mask, glob, n_full: int) -> np.ndarray:
    import torch
    n = len(tokens)
    acc = np.zeros((n, 3))
    cnt = np.zeros(n)
    with torch.no_grad():
        for s in _windows(n):
            sl = slice(s, s + WINDOW)
            p = torch.softmax(model(*(torch.from_numpy(v[sl][None]) for v in (tokens, mask, glob)))[0], -1)[0].numpy()
            st, sg, _ = _swap(tokens[sl], glob[sl], np.zeros(1))
            q = torch.softmax(model(torch.from_numpy(st[None]), torch.from_numpy(mask[sl][None]),
                                    torch.from_numpy(sg[None]))[0], -1)[0].numpy()
            acc[sl] += (p + q[:, [1, 0, 2]]) / 2
            cnt[sl] += 1
    p5 = acc / np.maximum(cnt, 1)[:, None]
    full = np.repeat(p5, STEP, axis=0)[:n_full]
    if len(full) < n_full:
        full = np.vstack([full, np.repeat(full[-1:], n_full - len(full), axis=0)])
    return full / full.sum(1, keepdims=True)


def seq_inputs(match, gt=None, clean=None):
    """Entradas a 5 fps de un partido (degradado) y, para entrenar, etiquetas y balon limpio."""
    tokens, mask, glob = frame_tokens(match)
    sl = slice(0, None, STEP)
    out = [tokens[sl], mask[sl], glob[sl]]
    if gt is not None:
        out.append(gt[sl])
        ball = np.column_stack([clean.ball_x / clean.pitch_length - 0.5, clean.ball_y / clean.pitch_width - 0.5])
        out.append(ball[sl].astype(np.float32))
    return out

"""Poseedor individual tipo PathCRF / Ball Radar (M5a del plan completo, E078).

PathCRF (Kim et al. 2026) y Ball Radar (Kim et al. 2023) no clasifican el equipo directamente: primero
estiman QUE JUGADOR tiene el balon y el equipo sale de sumar. Aqui, sobre el modelo profundo de E034
(atencion entre los jugadores visibles de cada frame + BiGRU de 10 s a 5 fps):
  - cabeza por jugador: logit de "este jugador controla el balon", mas un logit "nadie" (balon en
    vuelo o muerto) que sale del contexto del frame;
  - cabeza H/A/D como en seq.py.
El objetivo del poseedor sale del banco LIMPIO: en cada frame vivo, el jugador del equipo de la GT
mas cercano al balon limpio, si esta a < CTRL_DIST, se busca entre los tokens del partido degradado
(el slot a < 1,5 m de su posicion). P(H) y P(A) del modelo combinan la cabeza H/A/D con la masa de
probabilidad del poseedor de cada equipo. Prediccion simetrizada (equipos intercambiados).
"""

from __future__ import annotations

import numpy as np

from hackbarca.models.seq import MAX_TOKENS, N_GLOBAL_FEATS, N_TOKEN_FEATS, SEED, STEP, WINDOW, _windows

CTRL_DIST = 2.0      # m: el mas cercano del equipo poseedor al balon limpio "controla"
MATCH_DIST = 1.5     # m: emparejar ese jugador con un token del partido degradado


def tokens_with_order(match):
    """Como seq.frame_tokens pero devuelve tambien el orden de slots de cada token (n, MAX_TOKENS)."""
    from hackbarca.models.seq import frame_tokens
    tokens, mask, glob = frame_tokens(match)
    bx, by = match.ball_x, match.ball_y
    known = ~np.isnan(bx)
    d = np.hypot(match.players_x - np.nan_to_num(bx)[:, None], match.players_y - np.nan_to_num(by)[:, None])
    order = np.argsort(np.where(np.isnan(match.players_x), np.inf, np.where(known[:, None], d, 0.0)), axis=1)[:, :MAX_TOKENS]
    return tokens, mask, glob, order


def possessor_targets(clean, degraded, gt, order) -> np.ndarray:
    """Indice del token poseedor por frame (0..MAX_TOKENS-1), MAX_TOKENS = nadie, -1 = sin GT."""
    n = len(gt)
    out = np.full(n, MAX_TOKENS, dtype=np.int64)
    out[gt == 3] = -1
    live = (gt == 0) | (gt == 1)
    cbx, cby = clean.ball_x, clean.ball_y
    for team in (0, 1):
        cols = np.flatnonzero(clean.team_of_track == team)
        dist = np.hypot(clean.players_x[:, cols] - cbx[:, None], clean.players_y[:, cols] - cby[:, None])
        dist = np.where(np.isnan(dist), np.inf, dist)
        k = np.argmin(dist, axis=1)
        dk = dist[np.arange(n), k]
        sel = live & (gt == team) & np.isfinite(cbx) & (dk < CTRL_DIST)
        if not sel.any():
            continue
        f = np.flatnonzero(sel)
        tx, ty = clean.players_x[f, cols[k[f]]], clean.players_y[f, cols[k[f]]]
        slots = order[f]                                        # (m, MAX_TOKENS)
        sx = degraded.players_x[f[:, None], slots]
        sy = degraded.players_y[f[:, None], slots]
        same = degraded.team_of_track[slots] == team
        dd = np.where(same & ~np.isnan(sx), np.hypot(sx - tx[:, None], sy - ty[:, None]), np.inf)
        best = np.argmin(dd, axis=1)
        ok = dd[np.arange(len(f)), best] < MATCH_DIST
        out[f[ok]] = best[ok]
    return out


def build_model():
    import torch
    from torch import nn

    class Possessor(nn.Module):
        def __init__(self, d: int = 64, heads: int = 4):
            super().__init__()
            self.embed = nn.Sequential(nn.Linear(N_TOKEN_FEATS, d), nn.ReLU(), nn.Linear(d, d))
            layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.1, batch_first=True)
            self.attn = nn.TransformerEncoder(layer, num_layers=2)
            self.glob = nn.Linear(N_GLOBAL_FEATS, d)
            self.gru = nn.GRU(4 * d, d, batch_first=True, bidirectional=True)
            self.cls = nn.Linear(2 * d, 3)
            self.tok = nn.Sequential(nn.Linear(d + 2 * d, d), nn.ReLU(), nn.Linear(d, 1))
            self.none = nn.Linear(2 * d, 1)

        def forward(self, tokens, mask, glob):
            b, t, k, f = tokens.shape
            x = self.embed(tokens.reshape(b * t, k, f))
            m = mask.reshape(b * t, k).clone()
            m[~m.any(1), 0] = True
            x = self.attn(x, src_key_padding_mask=~m)
            team = tokens.reshape(b * t, k, f)[..., 4]
            pooled = []
            for sign in (1.0, -1.0):
                w = (m & (team == sign)).float().unsqueeze(-1)
                pooled.append((x * w).sum(1) / w.sum(1).clamp(min=1.0))
            allm = m.float().unsqueeze(-1)
            pooled.append((x * allm).sum(1) / allm.sum(1).clamp(min=1.0))
            pooled.append(self.glob(glob.reshape(b * t, -1)))
            z, _ = self.gru(torch.cat(pooled, -1).reshape(b, t, -1))       # (b, t, 2d)
            ctx = z.reshape(b * t, 1, -1).expand(-1, k, -1)
            tok_logit = self.tok(torch.cat([x, ctx], -1)).squeeze(-1)       # (b*t, k)
            tok_logit = tok_logit.masked_fill(~m, -1e4)
            none_logit = self.none(z.reshape(b * t, -1))                    # (b*t, 1)
            poss = torch.cat([tok_logit, none_logit], -1).reshape(b, t, k + 1)
            return self.cls(z), poss

    return Possessor()


def _swap(tokens, glob, y):
    t = tokens.copy()
    t[..., 4] *= -1
    yy = np.where(y == 0, 1, np.where(y == 1, 0, y))
    return t, glob, yy


def train_possessor(games, epochs: int = 6, device: str | None = None, w_poss: float = 1.0):
    """games: [(tokens, mask, glob, labels 0-3, poss_target)] a 5 fps."""
    import torch
    from torch import nn
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    items = []
    for tokens, mask, glob, y, poss in games:
        for s in _windows(len(y)):
            items.append(tuple(v[s:s + WINDOW] for v in (tokens, mask, glob, y, poss)))
    ce = nn.CrossEntropyLoss(ignore_index=3)
    ce_poss = nn.CrossEntropyLoss(ignore_index=-1)
    for _ in range(epochs):
        order = rng.permutation(len(items))
        for a in range(0, len(order), 64):
            batch = [items[i] for i in order[a:a + 64]]
            tk, mk, gl, yy, pp = (np.stack(c) for c in zip(*batch))
            if rng.random() < 0.5:
                tk, gl, yy = _swap(tk, gl, yy)     # el poseedor es el mismo token: no cambia
            tk, mk, gl = (torch.from_numpy(v).to(device) for v in (tk, mk, gl))
            logits, poss = model(tk, mk, gl)
            loss = ce(logits.reshape(-1, 3), torch.from_numpy(yy.astype(np.int64)).to(device).reshape(-1))
            loss = loss + w_poss * ce_poss(poss.reshape(-1, poss.shape[-1]), torch.from_numpy(pp).to(device).reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.to("cpu").eval()


def predict_possessor(model, tokens, mask, glob, n_full: int) -> np.ndarray:
    """P(H/A/D) a 25 fps: la cabeza H/A/D con el reparto H/A corregido por la masa del poseedor."""
    import torch
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        n = len(tokens)
        acc = np.zeros((n, 3))
        cnt = np.zeros(n)
        team = tokens[..., 4]
        with torch.no_grad():
            for s in _windows(n):
                sl = slice(s, s + WINDOW)
                outs = []
                for swap in (False, True):
                    tk, gl = (_swap(tokens[sl], glob[sl], np.zeros(1))[:2] if swap else (tokens[sl], glob[sl]))
                    logits, poss = model(torch.from_numpy(tk[None]), torch.from_numpy(mask[sl][None]),
                                         torch.from_numpy(gl[None]))
                    p = torch.softmax(logits, -1)[0].numpy()
                    q = torch.softmax(poss, -1)[0].numpy()             # (t, k+1)
                    tm = tk[..., 4]
                    mh = (q[:, :-1] * (tm == 1.0)).sum(1)
                    ma = (q[:, :-1] * (tm == -1.0)).sum(1)
                    live = p[:, 0] + p[:, 1]
                    share = np.where(mh + ma > 1e-6, mh / np.maximum(mh + ma, 1e-6), p[:, 0] / np.maximum(live, 1e-6))
                    # mezcla: la mitad del reparto H/A viene de la cabeza H/A/D y la mitad del poseedor
                    ph = live * (0.5 * p[:, 0] / np.maximum(live, 1e-6) + 0.5 * share)
                    out = np.column_stack([ph, live - ph, p[:, 2]])
                    outs.append(out[:, [1, 0, 2]] if swap else out)
                acc[sl] += (outs[0] + outs[1]) / 2
                cnt[sl] += 1
        del team
    finally:
        torch.use_deterministic_algorithms(previous)
    p5 = acc / np.maximum(cnt, 1)[:, None]
    full = np.repeat(p5, STEP, axis=0)[:n_full]
    if len(full) < n_full:
        full = np.vstack([full, np.repeat(full[-1:], n_full - len(full), axis=0)])
    return full / full.sum(1, keepdims=True)

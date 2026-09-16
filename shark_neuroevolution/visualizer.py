"""Draw the brain: sensors left, hidden mid, steering right. |w| = pink width."""
import numpy as np
import pygame

import config

PINK, GRAY, DIM = (255, 70, 180), (150, 150, 150), (42, 42, 52)
BG, TXT, FAINT = (8, 8, 14), (210, 210, 220), (110, 110, 125)

_fonts = {}  # module cache: SysFont does disk lookup, never build per-frame


def font(size):
    if size not in _fonts:
        _fonts[size] = pygame.font.SysFont(None, size)
    return _fonts[size]


def draw(surf, brain, obs, out=None):
    W, H = surf.get_size()
    surf.fill(BG)
    n_in, n_hid, n_out = config.INPUT_NODES, config.HIDDEN_NODES, config.OUTPUT_NODES
    w1, b1, w2, _ = brain.get_weights()  # w1: [hid, in], w2: [out, hid]

    f = font(20)
    surf.blit(f.render("LIVE BRAIN", True, TXT), (12, 8))
    strong = int(((np.abs(w1) > 0.5).sum() + (np.abs(w2) > 0.5).sum()))
    surf.blit(font(18).render(f"{strong} strong links", True, PINK), (12, 30))

    xs = [W * 0.16, W * 0.50, W * 0.84]
    top = 66  # reserve room for title + column labels
    for x, label in zip(xs, ("SENSORS", "HIDDEN", "STEERING")):
        t = f.render(label, True, FAINT)
        surf.blit(t, (x - t.get_width() / 2, top - 24))

    def ys(n):
        gap = min(30, (H - top - 16) / max(n, 1))
        return [top + 8 + i * gap for i in range(n)]

    pin, ph, po = [(xs[0], y) for y in ys(n_in)], [(xs[1], y) for y in ys(n_hid)], \
                  [(xs[2], y) for y in ys(n_out)]

    def edge(a, b, w):  # |w| -> pink + thick, ~0 -> faint gray
        m = min(1.0, abs(float(w)) * 2)
        color = tuple(int(DIM[i] + (PINK[i] - DIM[i]) * m) for i in range(3))
        pygame.draw.line(surf, color, a, b, 1 + int(m * 4))

    for i, a in enumerate(pin):
        if i < w1.shape[1]:
            for j, b in enumerate(ph):
                edge(a, b, w1[j, i])
    for j, a in enumerate(ph):
        for k, b in enumerate(po):
            edge(a, b, w2[k, j] if k < w2.shape[0] else 0)

    def node(p, r, color, glow=0.0):  # glow ring on highly active nodes
        if glow > 0.7:
            pygame.draw.circle(surf, PINK, (int(p[0]), int(p[1])), r + 3, 1)
        c = tuple(min(255, v + int(glow * 90)) for v in color)
        pygame.draw.circle(surf, c, (int(p[0]), int(p[1])), r)

    for i, p in enumerate(pin):
        node(p, 8, GRAY, min(1, abs(float(obs[i])) if i < len(obs) else 0))
    hid = np.maximum(0, w1 @ obs[:w1.shape[1]] + b1)  # mirror ReLU for display
    for j, p in enumerate(ph):
        node(p, 8, GRAY, min(1, float(hid[j])))
    for k, p in enumerate(po):  # outputs always pink; value shown in bars below
        v = float(out[k]) if out is not None and k < len(out) else 0.0
        node(p, 13, PINK, min(1, abs(v)))

    if out is not None:  # steering bars: ax / ay in [-1, 1]
        bw, bh = W * 0.10, 8
        for k, p in enumerate(po):
            if k >= len(out):
                break
            v = max(-1, min(1, float(out[k])))
            x0 = p[0] - bw / 2
            y0 = p[1] + 20 + k * 0  # stacked under each output node
            pygame.draw.rect(surf, DIM, (x0, y0, bw, bh), border_radius=4)
            cx = x0 + bw / 2  # fill grows left/right from center
            pygame.draw.rect(surf, PINK,
                            (cx, y0, (bw / 2) * v, bh) if v >= 0
                            else (cx + (bw / 2) * v, y0, (bw / 2) * -v, bh),
                            border_radius=4)
            t = font(16).render(f"{'ax' if k == 0 else 'ay'} {v:+.2f}", True, TXT)
            surf.blit(t, (x0 + bw / 2 - t.get_width() / 2, y0 + bh + 2))

    # legend, bottom-left
    ly = H - 18
    pygame.draw.line(surf, PINK, (14, ly), (44, ly), 5)
    surf.blit(font(16).render("strong", True, FAINT), (48, ly - 8))
    pygame.draw.line(surf, DIM, (120, ly), (150, ly), 1)
    surf.blit(font(16).render("weak", True, FAINT), (154, ly - 8))

"""Portrait UI: brain diagram (labeled growing inputs) + arena + HUD + caption."""
import math

import numpy as np
import pygame

import config

_fonts = {}  # SysFont does disk lookup: never build per-frame


def font(size, bold=False):
    if (size, bold) not in _fonts:
        _fonts[(size, bold)] = pygame.font.SysFont("arial", size, bold=bold)
    return _fonts[(size, bold)]


def _edges(surf, pairs):
    """pairs: (weight, a, b). Weak first so strong pink lines sit on top."""
    for wgt, a, b in sorted(pairs, key=lambda e: abs(float(e[0]))):
        m = min(1.0, abs(float(wgt)) * 3)
        if m < 0.12:  # background hair: barely-there, never competing
            pygame.draw.line(surf, (30, 30, 42), a, b, 1)
        else:
            color = tuple(int(config.DIM_GRAY[i] + (config.PINK[i] - config.DIM_GRAY[i]) * m)
                          for i in range(3))
            t = 1 + int(m * 3)
            if m > 0.55:  # highways get a halo pass
                pygame.draw.line(surf, color, a, b, t + 2)
            pygame.draw.line(surf, color, a, b, t)


def draw_network(surf, brain, obs, level_idx, out=0.0, new_inputs=0, fresh=False):
    """new_inputs: tail inputs brand-new this level; fresh: fill them pink for the
    first replays so viewers SEE the new senses. out: steering value for the bar."""
    surf.fill(config.BG_COLOR)
    lvl = config.LEVELS[level_idx]
    names = lvl["inputs"]
    w = brain.get_weights()
    w1, b1, w2 = w["w1"], w["b1"], w["w2"]
    W, H = surf.get_size()
    n_hid = config.HIDDEN_NODES

    t = font(20, True).render(
        f"Level {level_idx + 1} / {len(config.LEVELS)} \u00b7 {lvl['name']}",
        True, config.PINK)
    surf.blit(t, t.get_rect(center=(W // 2, 16)))

    ix, hx, ox = 150, W * 0.62, W * 0.88
    top0, bot = 74, H - 34  # legend strip reserved at bottom
    gap = min(30, (bot - top0) / max(len(names), 1))
    top = top0 + ((bot - top0) - (len(names) - 1) * gap) / 2  # block vertically centered
    pin = [(ix, top + i * gap) for i in range(len(names))]
    mid = top + (len(names) - 1) * gap / 2
    hgap = min(30, (bot - top0) / n_hid)
    ph = [(hx, mid + (i - (n_hid - 1) / 2) * hgap) for i in range(n_hid)]
    po = (ox, mid)

    for x, label in ((ix, "SENSORS"), (hx, "BRAIN"), (ox, "TURN")):  # column headers
        h = font(14).render(label, True, config.TEXT_GRAY)
        surf.blit(h, h.get_rect(center=(x, 48)))

    pairs = []  # input -> hidden
    for i, a in enumerate(pin):
        if i < w1.shape[1]:
            for j, b in enumerate(ph):
                pairs.append((w1[j, i], a, b))
    for j, a in enumerate(ph):  # hidden -> output
        pairs.append((w2[0, j] if j < w2.shape[1] else 0, a, po))
    _edges(surf, pairs)

    for i, p in enumerate(pin):  # labeled input nodes (pink ring = live)
        v = min(1, abs(float(obs[i]))) if i < len(obs) else 0
        pygame.draw.circle(surf, config.BG_COLOR, (int(p[0]), int(p[1])), 8)
        if fresh and new_inputs and i >= len(pin) - new_inputs:  # brand-new sense
            pygame.draw.circle(surf, config.PINK, (int(p[0]), int(p[1])), 8)
        else:
            pygame.draw.circle(surf, config.PINK, (int(p[0]), int(p[1])), 8,
                               2 if v > 0.05 else 1)
            if v > 0.7:
                pygame.draw.circle(surf, config.PINK, (int(p[0]), int(p[1])), 11, 1)
        lab = font(15).render(names[i], True,
                              config.PINK if v > 0.05 else config.TEXT_GRAY)
        surf.blit(lab, lab.get_rect(right=p[0] - 14, centery=p[1]))

    hid = np.maximum(0, w1 @ np.asarray(obs[:w1.shape[1]], dtype=np.float32) + b1)
    hm = float(hid.max()) + 1e-6  # hidden nodes light up with activation
    for j, p in enumerate(ph):
        g = min(1, float(hid[j]) / hm)
        c = tuple(int(config.DIM_GRAY[k] + (config.PINK[k] - config.DIM_GRAY[k]) * g)
                  for k in range(3))
        pygame.draw.circle(surf, c, (int(p[0]), int(p[1])), 6)

    v = max(-1, min(1, float(out)))  # output: filled pink + live turn bar
    pygame.draw.circle(surf, config.PINK, (int(po[0]), int(po[1])), 13, 1)
    pygame.draw.circle(surf, config.PINK, (int(po[0]), int(po[1])), 9)
    bw, bh, bx, by = 92, 8, po[0] - 46, po[1] + 22
    pygame.draw.rect(surf, (30, 30, 42), (bx, by, bw, bh), border_radius=4)
    cx = bx + bw / 2
    pygame.draw.rect(surf, config.PINK,
                     (cx, by, bw / 2 * v, bh) if v >= 0 else
                     (cx + bw / 2 * v, by, bw / 2 * -v, bh), border_radius=4)
    pygame.draw.line(surf, config.TEXT_GRAY, (cx, by - 2), (cx, by + bh + 2), 1)
    s = font(14).render(f"{v:+.2f}", True, config.TEXT_WHITE)
    surf.blit(s, s.get_rect(center=(cx, by + bh + 12)))

    ly = H - 12  # legend
    pygame.draw.line(surf, config.PINK, (14, ly), (40, ly), 4)
    surf.blit(font(13).render("strong", True, config.TEXT_GRAY), (44, ly - 7))
    pygame.draw.line(surf, (30, 30, 42), (110, ly), (136, ly), 1)
    surf.blit(font(13).render("weak", True, config.TEXT_GRAY), (140, ly - 7))


def draw_arena(surf, arena):
    ox, oy = config.GAME_RECT[0], config.GAME_RECT[1]
    surf.fill(config.BG_COLOR)
    pygame.draw.rect(surf, config.ARENA_COLOR, (0, 0, config.ARENA_W, config.ARENA_H),
                     border_radius=8)
    s = arena.shark
    if len(s.trail) > 1:  # fading trail
        pts = s.trail
        for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
            al = i / max(len(pts) - 1, 1)
            pygame.draw.line(surf, (40, int(40 + 60 * al), int(70 + 60 * al)), a, b, 3)
    for f in arena.fish:  # fish: ellipse body + tail fin
        if not f.alive:
            continue
        pygame.draw.ellipse(surf, config.FISH_COLOR, (f.x - 6, f.y - 3, 12, 6))
        td = -1 if f.vx > 0 else 1
        pygame.draw.polygon(surf, config.FISH_COLOR,
                            [(f.x + 6 * td, f.y), (f.x + 12 * td, f.y - 5),
                             (f.x + 12 * td, f.y + 5)])
    if arena.flash:  # eat burst: expanding pink ring
        pygame.draw.circle(surf, config.PINK, (int(s.x), int(s.y)),
                           config.SHARK_RADIUS + (20 - arena.flash) * 2, 2)
    a = s.angle  # shark triangle + pink glow ring
    pygame.draw.polygon(surf, config.SHARK_COLOR,
                        [(s.x + math.cos(a) * 18, s.y + math.sin(a) * 18),
                         (s.x + math.cos(a + 2.5) * 12, s.y + math.sin(a + 2.5) * 12),
                         (s.x + math.cos(a - 2.5) * 12, s.y + math.sin(a - 2.5) * 12)])
    pygame.draw.circle(surf, config.PINK, (int(s.x), int(s.y)), 22, 1)
    left = sum(f.alive for f in arena.fish)
    logo = font(14).render("ree[g]orithm", True, config.TEXT_WHITE)  # dark pills so
    pygame.draw.rect(surf, (0, 0, 0), logo.get_rect(topleft=(6, 4)).inflate(8, 4),
                     border_radius=4)  # fish never swim over the HUD text
    surf.blit(logo, (10, 8))
    t = font(14).render(f"{left} / {config.NUM_FISH}\nFISH LEFT", True, config.TEXT_WHITE)
    r = t.get_rect(topright=(config.ARENA_W - 10, 8)).inflate(8, 4)
    pygame.draw.rect(surf, (0, 0, 0), r, border_radius=4)
    surf.blit(t, t.get_rect(topright=(config.ARENA_W - 10, 8)))
    _ = (ox, oy)  # drawn into subsurface: arena coords already local


def draw_caption(surf, text):
    t = font(20, True).render(text, True, config.TEXT_WHITE)
    bg = t.get_rect(center=(config.WIDTH // 2, config.CAPTION_Y)).inflate(20, 10)
    pygame.draw.rect(surf, (0, 0, 0), bg, border_radius=4)
    surf.blit(t, t.get_rect(center=(config.WIDTH // 2, config.CAPTION_Y)))
    h = font(14).render("SPACE skip replay", True, config.TEXT_GRAY)
    surf.blit(h, h.get_rect(center=(config.WIDTH // 2, config.HEIGHT - 16)))

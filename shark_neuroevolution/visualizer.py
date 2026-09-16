"""Portrait UI: brain diagram (labeled growing inputs) + arena + HUD + caption."""
import math

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
    """Reference look: rings + labels only. New senses glow pink (label + ring),
    old ones rest gray. Two-tone level line sits under the graph."""
    surf.fill(config.BG_COLOR)
    lvl = config.LEVELS[level_idx]
    names = lvl["inputs"]
    w = brain.get_weights()
    w1, w2 = w["w1"], w["w2"]
    W, H = surf.get_size()
    n_hid = config.HIDDEN_NODES

    ix, hx, ox = 130, 265, 415  # centered fan with airy margins
    top0, bot = 80, H - 64
    gap = min(28, (bot - top0) / max(len(names), 1))
    top = top0 + ((bot - top0) - (len(names) - 1) * gap) / 2
    pin = [(ix, top + i * gap) for i in range(len(names))]
    mid = top + (len(names) - 1) * gap / 2
    hgap = min(28, (bot - top0) / n_hid)
    ph = [(hx, mid + (i - (n_hid - 1) / 2) * hgap) for i in range(n_hid)]
    po = (ox, mid)

    pairs = []  # input -> hidden
    for i, a in enumerate(pin):
        if i < w1.shape[1]:
            for j, b in enumerate(ph):
                pairs.append((w1[j, i], a, b))
    for j, a in enumerate(ph):  # hidden -> output
        pairs.append((w2[0, j] if j < w2.shape[1] else 0, a, po))
    _edges(surf, pairs)

    n_new = len(pin) - new_inputs if fresh and new_inputs else len(pin)
    for i, p in enumerate(pin):  # labeled rings; newest senses pink
        new = i >= n_new
        v = min(1, abs(float(obs[i]))) if i < len(obs) else 0
        c = config.PINK if (new or v > 0.5) else config.DIM_GRAY
        pygame.draw.circle(surf, c, (int(p[0]), int(p[1])), 6, 2)
        lab = font(15).render(names[i], True, c)
        surf.blit(lab, lab.get_rect(right=p[0] - 12, centery=p[1]))
    for p in ph:  # hidden rings, plain gray like the reference
        pygame.draw.circle(surf, config.DIM_GRAY, (int(p[0]), int(p[1])), 6, 1)
    pygame.draw.circle(surf, config.PINK, (int(po[0]), int(po[1])), 8, 2)

    a = font(20, True).render(f"Level {level_idx + 1} ", True, config.PINK)
    b = font(20, True).render(f"/ {len(config.LEVELS)} \u00b7 {lvl['name']}",
                              True, config.TEXT_WHITE)  # two-tone level line
    x = W / 2 - (a.get_width() + b.get_width()) / 2
    surf.blit(a, (x, H - 34))
    surf.blit(b, (x + a.get_width(), H - 34))


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
    for f in arena.fish:  # minnows: body + forked tail, oriented to velocity
        if not f.alive:
            continue
        ca, sa = math.cos(f.angle), math.sin(f.angle)
        rot = lambda px, py: (f.x + px * ca - py * sa, f.y + px * sa + py * ca)
        pygame.draw.polygon(surf, config.FISH_COLOR,
                            [rot(7, 0), rot(1, -2.5), rot(-5, -2),
                             rot(-5, 2), rot(1, 2.5)])
        pygame.draw.polygon(surf, config.FISH_COLOR,
                            [rot(-5, 0), rot(-10, -3.5), rot(-10, 3.5)])
    if arena.flash:  # eat burst: expanding pink ring
        pygame.draw.circle(surf, config.PINK, (int(s.x), int(s.y)),
                           config.SHARK_RADIUS + (20 - arena.flash) * 2, 2)
    a = s.angle  # shark silhouette: body + forked tail + dorsal + pectoral
    ca, sa = math.cos(a), math.sin(a)
    rot = lambda px, py: (s.x + px * ca - py * sa, s.y + px * sa + py * ca)
    pygame.draw.polygon(surf, config.SHARK_COLOR,  # tail flukes
                        [rot(-15, 0), rot(-27, -10), rot(-22, 0), rot(-27, 10)])
    pygame.draw.polygon(surf, config.SHARK_COLOR,  # body
                        [rot(24, 0), rot(10, -7), rot(-8, -8),
                         rot(-16, -3), rot(-16, 3), rot(-8, 8), rot(10, 7)])
    pygame.draw.polygon(surf, config.SHARK_COLOR,  # dorsal fin
                        [rot(-1, -7), rot(-8, -17), rot(-13, -7)])
    pygame.draw.polygon(surf, config.SHARK_COLOR,  # pectoral fin
                        [rot(5, 6), rot(-5, 15), rot(-3, 5)])
    pygame.draw.circle(surf, config.PINK, (int(s.x), int(s.y)), 26, 1)
    left = sum(f.alive for f in arena.fish)  # counter only: branding removed
    n = font(22, True).render(f"{left} / {config.NUM_FISH}", True, config.PINK)
    surf.blit(n, n.get_rect(topright=(config.ARENA_W - 12, 8)))
    t = font(13).render("FISH LEFT", True, config.TEXT_GRAY)
    surf.blit(t, t.get_rect(topright=(config.ARENA_W - 12, 32)))
    _ = (ox, oy)  # drawn into subsurface: arena coords already local


def draw_caption(surf, text):
    t = font(20, True).render(text, True, config.TEXT_WHITE)
    bg = t.get_rect(center=(config.WIDTH // 2, config.CAPTION_Y)).inflate(20, 10)
    pygame.draw.rect(surf, (0, 0, 0), bg, border_radius=4)
    surf.blit(t, t.get_rect(center=(config.WIDTH // 2, config.CAPTION_Y)))
    h = font(14).render("SPACE skip replay", True, config.TEXT_GRAY)
    surf.blit(h, h.get_rect(center=(config.WIDTH // 2, config.HEIGHT - 16)))

"""Night-hunt UI: gold-on-midnight brain rings + owl/mice arena. No branding."""
import math

import pygame

import config

A = config.ACCENT  # shorthand: learned / strong / new
HAIR = (28, 34, 58)  # background edges: visible, never competing

_fonts = {}  # SysFont does disk lookup: never build per-frame


def font(size, bold=False):
    if (size, bold) not in _fonts:
        _fonts[(size, bold)] = pygame.font.SysFont("arial", size, bold=bold)
    return _fonts[(size, bold)]


def _edges(surf, pairs):
    """pairs: (weight, a, b). Weak first so gold highways sit on top."""
    for wgt, a, b in sorted(pairs, key=lambda e: abs(float(e[0]))):
        m = min(1.0, abs(float(wgt)) * 3)
        if m < 0.12:
            pygame.draw.line(surf, HAIR, a, b, 1)
        else:
            color = tuple(int(config.DIM_GRAY[i] + (A[i] - config.DIM_GRAY[i]) * m)
                          for i in range(3))
            t = 1 + int(m * 3)
            if m > 0.55:  # highways get a halo pass
                pygame.draw.line(surf, color, a, b, t + 2)
            pygame.draw.line(surf, color, a, b, t)


def draw_network(surf, brain, obs, level_idx, new_inputs=0):
    """Rings + labels only. This level's own senses stay gold (like the reference);
    older ones glow gold only while their live signal is strong."""
    surf.fill(config.BG_COLOR)
    lvl = config.LEVELS[level_idx]
    names = lvl["inputs"]
    w = brain.get_weights()
    w1, w2 = w["w1"], w["w2"]
    W, H = surf.get_size()
    n_hid = config.HIDDEN_NODES

    ix, hx, ox = 130, 265, 415  # centered fan with airy margins
    top0, bot = 64, H - 64
    gap = min(28, (bot - top0) / max(len(names), 1))
    top = top0 + ((bot - top0) - (len(names) - 1) * gap) / 2
    pin = [(ix, top + i * gap) for i in range(len(names))]
    mid = top + (len(names) - 1) * gap / 2
    hgap = min(28, (bot - top0) / n_hid)
    ph = [(hx, mid + (i - (n_hid - 1) / 2) * hgap) for i in range(n_hid)]
    po = (ox, mid)

    pairs = []
    for i, a in enumerate(pin):
        if i < w1.shape[1]:
            for j, b in enumerate(ph):
                pairs.append((w1[j, i], a, b))
    for j, a in enumerate(ph):
        pairs.append((w2[0, j] if j < w2.shape[1] else 0, a, po))
    _edges(surf, pairs)

    first_new = len(pin) - new_inputs if new_inputs else len(pin)
    for i, p in enumerate(pin):
        v = min(1, abs(float(obs[i]))) if i < len(obs) else 0
        c = A if (i >= first_new or v > 0.5) else config.DIM_GRAY
        pygame.draw.circle(surf, c, (int(p[0]), int(p[1])), 6, 2)
        lab = font(15).render(names[i], True, c)
        surf.blit(lab, lab.get_rect(right=p[0] - 12, centery=p[1]))
    for p in ph:
        pygame.draw.circle(surf, config.DIM_GRAY, (int(p[0]), int(p[1])), 6, 1)
    pygame.draw.circle(surf, A, (int(po[0]), int(po[1])), 8, 2)

    a = font(20, True).render(f"Level {level_idx + 1} ", True, A)
    b = font(20, True).render(f"/ {len(config.LEVELS)} \u00b7 {lvl['name']}",
                              True, config.TEXT_WHITE)
    x = W / 2 - (a.get_width() + b.get_width()) / 2
    surf.blit(a, (x, H - 34))
    surf.blit(b, (x + a.get_width(), H - 34))


def draw_arena(surf, arena):
    surf.fill(config.BG_COLOR)
    pygame.draw.rect(surf, config.ARENA_COLOR, (0, 0, config.ARENA_W, config.ARENA_H),
                     border_radius=8)
    o = arena.owl
    if len(o.trail) > 1:  # fading flight trail
        pts = o.trail
        for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
            al = i / max(len(pts) - 1, 1)
            pygame.draw.line(surf, (30, 34 + int(50 * al), 70 + int(50 * al)), a, b, 3)
    for m in arena.mice:  # mice: body + ears + tail, oriented to heading
        if not m.alive:
            continue
        ca, sa = math.cos(m.angle), math.sin(m.angle)
        rot = lambda px, py: (m.x + px * ca - py * sa, m.y + px * sa + py * ca)
        pygame.draw.polygon(surf, config.MOUSE_COLOR,
                            [rot(7, 0), rot(2, -3), rot(-5, -2.5),
                             rot(-5, 2.5), rot(2, 3)])
        pygame.draw.circle(surf, config.MOUSE_COLOR, (int(rot(3, -3.5)[0]), int(rot(3, -3.5)[1])), 2)
        pygame.draw.circle(surf, config.MOUSE_COLOR, (int(rot(3, 3.5)[0]), int(rot(3, 3.5)[1])), 2)
        pygame.draw.line(surf, config.MOUSE_COLOR, rot(-5, 0), rot(-11, 3), 2)
    if arena.flash:  # catch burst: expanding gold ring
        pygame.draw.circle(surf, A, (int(o.x), int(o.y)),
                           config.OWL_RADIUS + (20 - arena.flash) * 2, 2)
    ca, sa = math.cos(o.angle), math.sin(o.angle)  # owl silhouette
    rot = lambda px, py: (o.x + px * ca - py * sa, o.y + px * sa + py * ca)
    pygame.draw.polygon(surf, config.OWL_COLOR,  # wings (tucked, not star-like)
                        [rot(-2, -10), rot(-14, -22), rot(-9, -8)])
    pygame.draw.polygon(surf, config.OWL_COLOR,
                        [rot(-2, 10), rot(-14, 22), rot(-9, 8)])
    pygame.draw.polygon(surf, config.OWL_COLOR,  # short wedge tail
                        [rot(-18, 0), rot(-27, -6), rot(-27, 6)])
    pygame.draw.polygon(surf, config.OWL_COLOR,  # body
                        [rot(22, 0), rot(9, -11), rot(-11, -12),
                         rot(-19, -4), rot(-19, 4), rot(-11, 12), rot(9, 11)])
    pygame.draw.polygon(surf, config.OWL_COLOR,  # beak
                        [rot(22, 0), rot(16, -3), rot(16, 3)])
    pygame.draw.circle(surf, config.BG_COLOR,  # dark eye reads on white head
                       (int(rot(10, -4)[0]), int(rot(10, -4)[1])), 3)
    pygame.draw.circle(surf, A, (int(o.x), int(o.y)), 28, 1)  # presence ring
    left = sum(m.alive for m in arena.mice)  # counter on quiet backing
    n = font(22, True).render(f"{left} / {config.NUM_MICE}", True, A)
    nr = n.get_rect(topright=(config.ARENA_W - 12, 8))
    pygame.draw.rect(surf, config.ARENA_COLOR, nr.inflate(12, 6), border_radius=4)
    surf.blit(n, nr)
    t = font(13).render("MICE LEFT", True, config.TEXT_GRAY)
    surf.blit(t, t.get_rect(topright=(config.ARENA_W - 12, 40)))


def draw_caption(surf, text):
    t = font(20, True).render(text, True, config.TEXT_WHITE)
    bg = t.get_rect(center=(config.WIDTH // 2, config.CAPTION_Y)).inflate(20, 10)
    pygame.draw.rect(surf, (0, 0, 0), bg, border_radius=4)
    surf.blit(t, t.get_rect(center=(config.WIDTH // 2, config.CAPTION_Y)))
    h = font(14).render("SPACE skip replay", True, config.TEXT_GRAY)
    surf.blit(h, h.get_rect(center=(config.WIDTH // 2, config.HEIGHT - 16)))

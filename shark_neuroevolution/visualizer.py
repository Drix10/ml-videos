"""Portrait UI: brain diagram (labeled growing inputs) + arena + HUD + caption."""
import math

import pygame

import config

_fonts = {}  # SysFont does disk lookup: never build per-frame


def font(size, bold=False):
    if (size, bold) not in _fonts:
        _fonts[(size, bold)] = pygame.font.SysFont("arial", size, bold=bold)
    return _fonts[(size, bold)]


def _edge(surf, a, b, w):  # |w| -> pink + thick, ~0 -> faint gray
    m = min(1.0, abs(float(w)) * 3)
    if m < 0.1:
        color, t = config.DIM_GRAY, 1
    else:
        color = tuple(int(config.DIM_GRAY[i] + (config.PINK[i] - config.DIM_GRAY[i]) * m)
                      for i in range(3))
        t = max(1, int(m * 3))
    pygame.draw.line(surf, color, a, b, t)


def draw_network(surf, brain, obs, level_idx):
    surf.fill(config.BG_COLOR)
    lvl = config.LEVELS[level_idx]
    names = lvl["inputs"]
    w = brain.get_weights()
    w1, w2 = w["w1"], w["w2"]
    W, H = surf.get_size()

    t = font(20, True).render(
        f"Level {level_idx + 1} / {len(config.LEVELS)} \u00b7 {lvl['name']}",
        True, config.PINK)
    surf.blit(t, t.get_rect(center=(W // 2, 14)))

    ix, hx, ox = W * 0.30, W * 0.62, W * 0.88  # room for labels left of inputs
    top, avail = 56, H - 70
    gap = min(34, avail / max(len(names), 1))
    pin = [(ix, top + i * gap) for i in range(len(names))]
    ph = [(hx, top + (i - (config.HIDDEN_NODES - 1) / 2) *
           min(34, avail / config.HIDDEN_NODES) + avail / 2 - avail / 2)
          for i in range(config.HIDDEN_NODES)]
    # center hidden column on inputs block
    mid = top + (len(names) - 1) * gap / 2
    hg = min(34, avail / config.HIDDEN_NODES)
    ph = [(hx, mid + (i - (config.HIDDEN_NODES - 1) / 2) * hg)
          for i in range(config.HIDDEN_NODES)]
    po = (ox, mid)

    for i, a in enumerate(pin):  # input -> hidden
        if i < w1.shape[1]:
            for j, b in enumerate(ph):
                _edge(surf, a, b, w1[j, i])
    for j, a in enumerate(ph):  # hidden -> output
        _edge(surf, a, po, w2[0, j] if j < w2.shape[1] else 0)

    for i, p in enumerate(pin):  # labeled input nodes (pink ring = live)
        v = min(1, abs(float(obs[i]))) if i < len(obs) else 0
        pygame.draw.circle(surf, config.BG_COLOR, (int(p[0]), int(p[1])), 8)
        pygame.draw.circle(surf, config.PINK, (int(p[0]), int(p[1])), 8, 2 if v > 0.05 else 1)
        if v > 0.7:
            pygame.draw.circle(surf, config.PINK, (int(p[0]), int(p[1])), 11, 1)
        lab = font(15).render(names[i], True,
                              config.PINK if v > 0.05 else config.TEXT_GRAY)
        surf.blit(lab, lab.get_rect(right=p[0] - 14, centery=p[1]))
    for p in ph:  # hidden nodes
        pygame.draw.circle(surf, config.DIM_GRAY, (int(p[0]), int(p[1])), 6, 1)
    pygame.draw.circle(surf, config.PINK, (int(po[0]), int(po[1])), 9, 2)  # output


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
    surf.blit(font(14).render("ree[g]orithm", True, config.TEXT_WHITE), (10, 8))
    t = font(14).render(f"{left} / {config.NUM_FISH}\nFISH LEFT", True, config.TEXT_WHITE)
    surf.blit(t, t.get_rect(topright=(config.ARENA_W - 10, 8)))
    _ = (ox, oy)  # drawn into subsurface: arena coords already local


def draw_caption(surf, text):
    t = font(20, True).render(text, True, config.TEXT_WHITE)
    bg = t.get_rect(center=(config.WIDTH // 2, config.CAPTION_Y)).inflate(20, 10)
    pygame.draw.rect(surf, (0, 0, 0), bg, border_radius=4)
    surf.blit(t, t.get_rect(center=(config.WIDTH // 2, config.CAPTION_Y)))
    h = font(14).render("SPACE skip replay", True, config.TEXT_GRAY)
    surf.blit(h, h.get_rect(center=(config.WIDTH // 2, config.HEIGHT - 16)))

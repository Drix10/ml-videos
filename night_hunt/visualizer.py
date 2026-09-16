"""Night-hunt UI: gold-on-midnight, rendered at UI_SCALE for crisp footage.
All geometry is authored in logical units (540x860) and scaled at draw time;
stroke widths stay hairline so lines never go fluffy."""
import math

import pygame

import config

S = config.UI_SCALE
A = config.ACCENT  # shorthand: learned / strong / new
HAIR = (28, 34, 58)  # background edges: visible, never competing

_DISPLAY = ("segoe ui", "arial")  # premium system faces: zero downloads
_MONO = ("consolas", "courier new")
_fonts = {}


def font(size, bold=False, mono=False):
    size = int(size * S)  # logical pt -> device px
    key = (size, bold, mono)  # SysFont does disk lookup: never build per-frame
    if key not in _fonts:
        _fonts[key] = pygame.font.SysFont(
            _MONO[0] if mono else _DISPLAY[0], size, bold=bold)
    return _fonts[key]


_vig = {}  # vignettes, pre-rendered once per size


def _vignette(w, h):
    if (w, h) not in _vig:
        v = pygame.Surface((w, h), pygame.SRCALPHA)
        cx, cy, maxr = w / 2, h / 2, math.hypot(w / 2, h / 2)
        for r in range(int(maxr), 0, -4):
            pygame.draw.circle(v, (0, 0, 0, int(70 * (1 - r / maxr) ** 2)),
                               (int(cx), int(cy)), r)
        _vig[(w, h)] = v
    return _vig[(w, h)]


def _edges(surf, pairs, mul=1):
    """Weak first so gold highways sit on top. mul scales widths for SS passes."""
    for wgt, a, b in sorted(pairs, key=lambda e: abs(float(e[0]))):
        m = min(1.0, abs(float(wgt)) * 3)
        if m < 0.12:
            pygame.draw.line(surf, HAIR, a, b, max(1, mul))
        else:
            color = tuple(int(config.DIM_GRAY[i] + (A[i] - config.DIM_GRAY[i]) * m)
                          for i in range(3))
            pygame.draw.line(surf, color, a, b, max(1, (1 + int(m * 2)) * mul))


_seen = {}  # level_idx -> tick of first draw (slide-in animation)
_lab = {}  # label surfaces: (name, gold) -> cached render
_SS = {}  # geometry buffers keyed by panel size (supersampled for AA)
SS = 2  # geometry rendered 2x then downscaled: lines come out anti-aliased


def draw_network(surf, brain, obs, level_idx, new_inputs=0):
    """Rings + labels only. This level's own senses stay gold (like the reference);
    older ones glow gold only while their live signal is strong."""
    surf.fill(config.BG_COLOR)
    lvl = config.LEVELS[level_idx]
    names = lvl["inputs"]
    w = brain.get_weights()
    w1, w2 = w["w1"], w["w2"]
    n_hid = config.HIDDEN_NODES

    ix, hx, ox = 130, 265, 415  # centered fan, logical units
    top0, bot = 64, surf.get_size()[1] // S - 64
    gap = min(28, (bot - top0) / max(len(names), 1))
    top = top0 + ((bot - top0) - (len(names) - 1) * gap) / 2
    pin = [(ix, top + i * gap) for i in range(len(names))]
    sliding = False
    if new_inputs:  # new senses slide in from the left once per level
        now0 = pygame.time.get_ticks()
        if level_idx not in _seen:
            _seen[level_idx] = now0
        p_ = min(1.0, (now0 - _seen[level_idx]) / 450.0)
        sliding = p_ < 1.0
        ease = 1 - (1 - p_) ** 3
        first = len(pin) - new_inputs
        pin = [((70 + (ix - 70) * ease) if i >= first else x, y)
               for i, (x, y) in enumerate(pin)]
    mid = top + (len(names) - 1) * gap / 2
    hgap = min(28, (bot - top0) / n_hid)
    ph = [(hx, mid + (i - (n_hid - 1) / 2) * hgap) for i in range(n_hid)]
    po = (ox, mid)

    P = lambda p: (p[0] * S, p[1] * S)  # logical -> device
    G = lambda p: (p[0] * S * SS, p[1] * S * SS)  # logical -> supersampled
    key = (surf.get_size(), id(brain))
    layer = None if sliding else _SS.get(key)
    if layer is None:  # static layer: edges + gray base rings, baked once/brain
        ss = _SS.get(surf.get_size())
        if ss is None:  # one reusable buffer per panel size (no per-frame alloc)
            ss = pygame.Surface((surf.get_size()[0] * SS, surf.get_size()[1] * SS))
            _SS[surf.get_size()] = ss
        ss.fill(config.BG_COLOR)
        pairs = []
        for i, a in enumerate(pin):
            if i < w1.shape[1]:
                for j, b in enumerate(ph):
                    pairs.append((w1[j, i], G(a), G(b)))
        for j, a in enumerate(ph):
            pairs.append((w2[0, j] if j < w2.shape[1] else 0, G(a), G(po)))
        _edges(ss, pairs, SS)
        for p in pin + ph:  # gray base rings for every node
            X, Y = G(p)
            pygame.draw.circle(ss, config.DIM_GRAY, (int(X), int(Y)),
                               6 * S * SS, SS)
        X, Y = G(po)
        pygame.draw.circle(ss, A, (int(X), int(Y)), 8 * S * SS, max(2, S) * SS)
        layer = pygame.Surface(surf.get_size())
        pygame.transform.smoothscale(ss, surf.get_size(), layer)
        if not sliding:  # weights frozen mid-replay: keep 2 zeitgeists max
            _SS[key] = layer
            baked = [k for k in _SS if isinstance(k, tuple) and len(k) == 2
                     and isinstance(k[0], tuple)]
            for k in baked[:-2]:
                del _SS[k]
    surf.blit(layer, (0, 0))
    first_new = len(pin) - new_inputs if new_inputs else len(pin)
    now = pygame.time.get_ticks()  # per frame: gold overlays on live nodes only
    for i, p in enumerate(pin):
        v = min(1, abs(float(obs[i]))) if i < len(obs) else 0
        if not (i >= first_new or v > 0.5):
            continue
        r = (6 + int(1.5 * math.sin(now * 0.006 + i * 0.8))) * S
        X, Y = P(p)
        pygame.draw.circle(surf, A, (int(X), int(Y)), r, max(1, S // 2 + 1))

    for i, p in enumerate(pin):  # text pass: fonts are already AA, draw direct
        v = min(1, abs(float(obs[i]))) if i < len(obs) else 0
        gold = i >= first_new or v > 0.5
        X, Y = P(p)
        lk = (names[i], gold)
        lab = _lab.get(lk)
        if lab is None:
            lab = font(15).render(names[i], True, A if gold else config.DIM_GRAY)
            _lab[lk] = lab
        surf.blit(lab, lab.get_rect(right=X - 12 * S, centery=Y))

    a = font(20, True).render(f"Level {level_idx + 1} ", True, A)
    b = font(20, True).render(f"/ {len(config.LEVELS)} \u00b7 {lvl['name']}",
                              True, config.TEXT_WHITE)
    W = surf.get_size()[0]
    x = W / 2 - (a.get_width() + b.get_width()) / 2
    surf.blit(a, (x, surf.get_size()[1] - 34 * S))
    surf.blit(b, (x + a.get_width(), surf.get_size()[1] - 34 * S))


def draw_arena(surf, arena):
    surf.fill(config.BG_COLOR)
    W, H = config.ARENA_W * S, config.ARENA_H * S
    pygame.draw.rect(surf, config.ARENA_COLOR, (0, 0, W, H), border_radius=8 * S)
    Q = lambda x, y: (x * S, y * S)  # physics -> device
    o = arena.owl
    if len(o.trail) > 1:  # fading flight trail
        pts = o.trail
        for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
            al = i / max(len(pts) - 1, 1)
            pygame.draw.line(surf, (30, 34 + int(50 * al), 70 + int(50 * al)),
                             Q(*a), Q(*b), 3)
    for m in arena.mice:
        if len(m.trail) > 1 and m.alive:  # faint motion ribbons
            pts = m.trail
            for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
                if i / max(len(pts) - 1, 1) > 0.4:
                    pygame.draw.line(surf, (56, 66, 104), Q(*a), Q(*b), 2)
    for m in arena.mice:  # mice: body + ears + tail, oriented to heading
        if not m.alive:
            continue
        ca, sa = math.cos(m.angle), math.sin(m.angle)
        rot = lambda px, py: Q(m.x + px * ca - py * sa, m.y + px * sa + py * ca)
        pygame.draw.polygon(surf, config.MOUSE_COLOR,
                            [rot(7, 0), rot(2, -3), rot(-5, -2.5),
                             rot(-5, 2.5), rot(2, 3)])
        ex, ey = rot(3, -3.5)
        pygame.draw.circle(surf, config.MOUSE_COLOR, (int(ex), int(ey)), 2 * S)
        ex, ey = rot(3, 3.5)
        pygame.draw.circle(surf, config.MOUSE_COLOR, (int(ex), int(ey)), 2 * S)
        pygame.draw.line(surf, config.MOUSE_COLOR, rot(-5, 0), rot(-11, 3), 2)
    if arena.flash:  # catch burst: expanding gold ring
        pygame.draw.circle(surf, A, (int(o.x * S), int(o.y * S)),
                           int((config.OWL_RADIUS + (20 - arena.flash) * 2) * S), 2)
    for s in arena.sparks:  # burst motes
        r = max(1, int(3 * s.life * S))
        mote = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(mote, (*A, int(255 * s.life)), (r, r), r)
        surf.blit(mote, (s.x * S - r, s.y * S - r))
    surf.blit(_vignette(W, H), (0, 0))
    ca, sa = math.cos(o.angle), math.sin(o.angle)  # owl silhouette
    rot = lambda px, py: Q(o.x + px * ca - py * sa, o.y + px * sa + py * ca)
    pygame.draw.polygon(surf, config.OWL_COLOR,  # tucked wings
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
    ex, ey = rot(10, -4)  # dark eye reads on white head
    pygame.draw.circle(surf, config.BG_COLOR, (int(ex), int(ey)), 3 * S)
    pygame.draw.circle(surf, A, (int(o.x * S), int(o.y * S)), 28 * S, 2)
    left = sum(m.alive for m in arena.mice)  # counter on quiet backing
    mark = font(14, True).render("drix10.com", True, config.TEXT_GRAY)
    surf.blit(mark, (12 * S, 8 * S))
    n = font(22, True, mono=True).render(f"{left} / {config.NUM_MICE}", True, A)
    nr = n.get_rect(topright=(W - 12 * S, 8 * S))
    pygame.draw.rect(surf, config.ARENA_COLOR, nr.inflate(12 * S, 6 * S),
                     border_radius=4 * S)
    surf.blit(n, nr)
    t = font(13).render("MICE LEFT", True, config.TEXT_GRAY)
    surf.blit(t, t.get_rect(topright=(W - 12 * S, 40 * S)))
    pygame.draw.rect(surf, config.ARENA_BORDER, (0, 0, W, H), 2,
                     border_radius=8 * S)  # bright outline, drawn last

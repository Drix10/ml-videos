"""Night-hunt UI: gold-on-midnight, rendered at UI_SCALE for crisp footage.
All geometry is authored in logical units (540x960) and scaled at draw time;
stroke widths stay hairline so lines never go fluffy."""
import math

import numpy as np
import pygame

import config

S = config.UI_SCALE
A = config.ACCENT  # shorthand: learned / strong / new
HAIR = (28, 34, 58)  # background edges: visible, never competing

try:
    from pygame import gfxdraw as _gfx  # AA rims for sprites (bundled)
except ImportError:
    _gfx = None


def _poly(surf, color, pts):
    """Filled polygon + AA rim: crisp sprites, no jagged shimmer."""
    pts = [(int(x), int(y)) for x, y in pts]
    pygame.draw.polygon(surf, color, pts)
    if _gfx is not None:
        _gfx.aapolygon(surf, pts, color)


def _dot(surf, color, xy, r):
    x, y = int(xy[0]), int(xy[1])
    pygame.draw.circle(surf, color, (x, y), r)
    if _gfx is not None and r >= 2:
        _gfx.aacircle(surf, x, y, r, color)


_DISPLAY = ("segoe ui", "arial")  # premium system faces: zero downloads
_MONO = ("consolas", "courier new")
_fonts = {}


def font(size, bold=False, mono=False):
    size = int(size * S)  # logical pt -> device px
    key = (size, bold, mono)  # SysFont does disk lookup: never build per-frame
    if key not in _fonts:
        # SysFont accepts a comma-separated name list and picks the first
        # available one — passing only [0] (as this did before) meant the
        # declared "arial"/"courier new" fallbacks were dead code: on any
        # box without Segoe UI/Consolas installed (non-Windows, or a
        # stripped-down Windows image), pygame would silently drop to its
        # own generic default font instead of the intended fallback face.
        names = ",".join(_MONO if mono else _DISPLAY)
        _fonts[key] = pygame.font.SysFont(names, size, bold=bold)
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


_seen = {}  # level_idx -> tick of first draw (slide-in animation)
_lab = {}  # label surfaces: (name, gold) -> cached render
_FLOW = {}  # brain id -> smoothed per-edge signals (temporal EMA)


def draw_network(surf, brain, obs, level_idx, new_inputs=0):
    """Every line live: per-edge signals smoothed frame-to-frame (EMA) so
    the web breathes colored<->dim instead of flickering. Constant hairline
    widths -- signal shows in COLOR only, which is what keeps cores crisp."""
    lvl = config.LEVELS[level_idx]
    names = lvl["inputs"]
    hit = _FLOW.get(id(brain))  # weights frozen mid-replay: fetch once,
    if hit is not None and hit[0] is brain:  # identity-guarded (id() lies
        _, r1, r2, prev, hist = hit  # after gc: a recycled id must not
    else:  # inherit another brain's smoothed signals. r1/r2: weight
        w = brain.get_weights()  # STRUCTURE relative per layer -- the
        a1, a2 = np.abs(w["w1"]), np.abs(w["w2"])  # skeleton the live
        r1 = a1 / a1.max() if a1.max() > 0 else a1  # signal dances on
        r2 = a2 / a2.max() if a2.max() > 0 else a2
        prev, hist = None, []
    n_hid = config.HIDDEN_NODES

    ix, hx, ox = 130, 265, 415  # centered fan, logical units
    top0, bot = 64, surf.get_size()[1] // S - 64
    gap = min(28, (bot - top0) / max(len(names), 1))
    top = top0 + ((bot - top0) - (len(names) - 1) * gap) / 2
    pin = [(ix, top + i * gap) for i in range(len(names))]
    if new_inputs:  # new senses slide in from the left once per level
        now0 = pygame.time.get_ticks()
        if level_idx not in _seen:
            _seen[level_idx] = now0
        p_ = min(1.0, (now0 - _seen[level_idx]) / 450.0)
        ease = 1 - (1 - p_) ** 3
        first = len(pin) - new_inputs
        pin = [((100 + (ix - 100) * ease) if i >= first else x, y)
               for i, (x, y) in enumerate(pin)]
    mid = top + (len(names) - 1) * gap / 2
    hgap = min(28, (bot - top0) / n_hid)
    ph = [(hx, mid + (i - (n_hid - 1) / 2) * hgap) for i in range(n_hid)]
    po = (ox, mid)

    P = lambda p: (p[0] * S, p[1] * S)  # logical -> device
    first_new = len(pin) - new_inputs if new_inputs else 0  # L1: the only
    # sense IS the new sense -- without this the whole L1 panel stays dark
    surf.fill(config.BG_COLOR)
    h_act, o_act = brain.activations(obs)  # this frame's live signal
    n_in = min(len(obs), r1.shape[1])
    n_h = min(n_hid, len(h_act), r2.shape[1])
    hm = max([float(x) for x in h_act[:n_h]], default=0.0)
    hist.append(np.array(obs[:n_in], float, copy=True))  # auto-exposure:
    del hist[:-120]  # a saturating sense (dist hovering 0.85-0.95) gets
    if len(hist) > 4:  # rescaled to its recent range -- same data, visible
        H = np.array(hist)  # swing. Truly flat (bias) keeps raw: nothing
        lo, span = H.min(axis=0), np.ptp(H, axis=0)  # to expose.
    else:
        lo, span = np.zeros(n_in), np.zeros(n_in)
    cur = {}  # structure x live: strong paths always stand, firing ones
    for i in range(n_in):  # flare -- the reference's highways, breathing.
        raw = min(1.0, abs(float(obs[i])))  # rescaled to its recent range
        v = min(1.0, max(0.0, (raw - lo[i]) / span[i])) if span[i] >= 0.05 \
            else raw  # truly flat (bias) keeps raw: nothing to expose
        for j in range(n_h):  # LOW floor (0.15): quiet frames genuinely
            cur[(0, i, j)] = r1[j, i] ** 0.5 * (0.15 + 0.85 * v)
    for j in range(n_h):  # a high floor + sqrt washed everything toward
        a = float(h_act[j]) / hm if hm > 1e-9 else 0.0  # gold and killed
        cur[(1, j, 0)] = r2[0, j] ** 0.5 * (0.15 + 0.85 * min(1.0, a))
    sm = dict(cur) if prev is None else {k: 0.2 * v + 0.8 * prev.get(k, 0.0)
                                         for k, v in cur.items()}
    _FLOW[id(brain)] = (brain, r1, r2, sm, hist)
    while len(_FLOW) > 2:  # only recent brains persist
        _FLOW.pop(next(iter(_FLOW)))
    for (kind, i, j), s in sorted(sm.items(), key=lambda kv: kv[1]):
        m = min(1.0, s)  # weak first so hot lines sit on top. Floor is a
        # VISIBLE gray web (never near-invisible): every path stays
        # connected like the reference, gold breathes on top of it.
        g = m ** 0.7  # COLOR ONLY: perceptual lift. Ordering untouched,
        # but mid-drives (0.1-0.3, where this brain's whole fan lives)
        # render warm instead of sinking into the gray -- the reference's
        # fan is graded the same way: a few hot, several warm, rest web.
        color = tuple(int(config.DIM_GRAY[k] + (A[k] - config.DIM_GRAY[k])
                          * g) for k in range(3))
        a_ = P(pin[i]) if kind == 0 else P(ph[i])  # kind 1 keys are
        b_ = P(ph[j]) if kind == 0 else P(po)  # (1, hidden, out=0): the
        # hidden index lives in i, NOT j -- j is the output slot (always 0),
        # so ph[j] stacked every follower under the top node's line
        pygame.draw.line(surf, color, (int(a_[0]), int(a_[1])),
                         (int(b_[0]), int(b_[1])), max(1, S))
    for p in pin + ph:  # quiet base rings under the live overlay
        X, Y = P(p)
        pygame.draw.circle(surf, config.DIM_GRAY, (int(X), int(Y)),
                           6 * S, max(1, S))
    for j, p in enumerate(ph):  # hot hidden nodes glow: relative --
        # absolute acts peak ~0.3 (measured), so a fixed 1.0 never fires
        if hm > 1e-9 and j < len(h_act) and float(h_act[j]) > 0.3 * hm:
            X, Y = P(p)
            pygame.draw.circle(surf, A, (int(X), int(Y)), 6 * S,
                               max(1, S) + S)
    X, Y = P(po)  # output ring widens with the live turn command
    pygame.draw.circle(surf, A, (int(X), int(Y)), 8 * S,
                       max(2, S) + int(abs(float(o_act)) * 2 * S))

    now = pygame.time.get_ticks()  # new input rings pulse all level
    for i, p in enumerate(pin):
        if i < first_new:
            continue
        r = (6 + int(1.5 * math.sin(now * 0.006 + i * 0.8))) * S
        X, Y = P(p)
        pygame.draw.circle(surf, A, (int(X), int(Y)), r, max(1, S // 2 + 1))

    for i, p in enumerate(pin):  # text pass: fonts are AA, draw direct
        X, Y = P(p)
        gold = i >= first_new  # the new sense glows all level; old ones rest
        lk = (names[i], gold)
        lab = _lab.get(lk)
        if lab is None:
            lab = font(15).render(names[i], True,
                                  A if gold else config.DIM_GRAY)
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
                             Q(*a), Q(*b), 3 * S)
    for m in arena.mice:
        if len(m.trail) > 1 and m.alive:  # faint motion ribbons
            pts = m.trail
            for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
                if i / max(len(pts) - 1, 1) > 0.4:
                    pygame.draw.line(surf, (56, 66, 104), Q(*a), Q(*b), 2 * S)
    for m in arena.mice:  # mice: body + ears + tail, oriented to heading
        if not m.alive:
            continue
        ca, sa = math.cos(m.angle), math.sin(m.angle)
        rot = lambda px, py: Q(m.x + px * ca - py * sa, m.y + px * sa + py * ca)
        _poly(surf, config.MOUSE_COLOR,
              [rot(7, 0), rot(2, -3), rot(-5, -2.5),
               rot(-5, 2.5), rot(2, 3)])
        _dot(surf, config.MOUSE_COLOR, rot(3, -3.5), 2 * S)
        _dot(surf, config.MOUSE_COLOR, rot(3, 3.5), 2 * S)
        pygame.draw.line(surf, config.MOUSE_COLOR, rot(-5, 0), rot(-11, 3),
                         2 * S)
    if arena.flash:  # catch burst: expanding gold ring
        pygame.draw.circle(surf, A, (int(o.x * S), int(o.y * S)),
                           int((config.OWL_RADIUS + (20 - arena.flash) * 2) * S),
                           2 * S)
    for s in arena.sparks:  # burst motes
        r = max(1, int(3 * s.life * S))
        mote = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(mote, (*A, int(255 * s.life)), (r, r), r)
        surf.blit(mote, (s.x * S - r, s.y * S - r))
    surf.blit(_vignette(W, H), (0, 0))
    ca, sa = math.cos(o.angle), math.sin(o.angle)  # owl silhouette
    rot = lambda px, py: Q(o.x + px * ca - py * sa, o.y + px * sa + py * ca)
    _poly(surf, config.OWL_COLOR,  # tucked wings
          [rot(-2, -10), rot(-14, -22), rot(-9, -8)])
    _poly(surf, config.OWL_COLOR,
          [rot(-2, 10), rot(-14, 22), rot(-9, 8)])
    _poly(surf, config.OWL_COLOR,  # short wedge tail
          [rot(-18, 0), rot(-27, -6), rot(-27, 6)])
    _poly(surf, config.OWL_COLOR,  # body
          [rot(22, 0), rot(9, -11), rot(-11, -12),
           rot(-19, -4), rot(-19, 4), rot(-11, 12), rot(9, 11)])
    _poly(surf, config.OWL_COLOR,  # beak
          [rot(22, 0), rot(16, -3), rot(16, 3)])
    _dot(surf, config.BG_COLOR, rot(10, -4), 3 * S)  # dark eye on white head
    pygame.draw.circle(surf, A, (int(o.x * S), int(o.y * S)), 28 * S,
                       max(2, 2 * S))
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
    pygame.draw.rect(surf, config.ARENA_BORDER, (0, 0, W, H), max(2, 2 * S),
                     border_radius=8 * S)  # bright outline, drawn last
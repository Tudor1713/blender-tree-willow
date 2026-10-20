# -*- coding: utf-8 -*-
"""
Procedural hero tree for Blender 5.2 (Cycles / HIP)
===================================================
  * Trunk  : geometric bark -- buttress roots, a wandering vertical fissure
             network, plate-scale noise, burls, and six broken dead limbs.
             UVs are in METRES so the procedural bark shader can add fine
             grain on top of the modelled relief.
  * Branch : recursive continuation + lateral forking, gentle gravity droop,
             foliage distributed along the WHOLE limb (not just at the tips).
  * Leaf   : every leaf is an individually modelled blade (22 verts) with a
             cupped cross-section, a raised midrib and four pairs of lateral
             vein ridges.  ~40k leaves merged into one mesh, each carrying
             its own vertex colour (hue / value variation) and a random
             attribute for shader-level variation.
  * Light  : low golden-hour sun behind the crown -> rim-lit, translucent,
             glowing foliage; cool sky fill from the camera side; soft ground
             bounce; optional bounded mist volume.
"""

import bpy
import math
import random
import colorsys
import os
import sys
import time
from array import array
from mathutils import Vector, Matrix, Quaternion
from mathutils import noise as mnoise

# --------------------------------------------------------------------------
# CLI  (Blender swallows the "--" separator -> scan every argument)
# --------------------------------------------------------------------------
RAW = [a for a in sys.argv[1:]]


def arg(name, default=None):
    for i, a in enumerate(RAW):
        if a == name and i + 1 < len(RAW):
            return RAW[i + 1]
    return default


def flag(name):
    return name in RAW


OUT_DIR = arg("--out", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tree_output"))
SEED = int(arg("--seed", 20261020))
RENDER_W = int(arg("--rw", 2000))
RENDER_H = int(arg("--rh", 2500))
SAMPLES = int(arg("--spp", 256))
DO_RENDER = not flag("--norender")
CAM_NAME = arg("--cam", "hero")
USE_MIST = not flag("--nomist")
SAVE_BLEND = not flag("--nosave")
MIST_DENSITY = float(arg("--mist", 0.0075))

os.makedirs(OUT_DIR, exist_ok=True)
T0 = time.time()


def log(msg):
    print("[tree] %7.1fs  %s" % (time.time() - T0, msg), flush=True)


# --------------------------------------------------------------------------
# Noise helpers
# --------------------------------------------------------------------------
def fbm(v, octaves=4, lac=2.0, gain=0.5):
    tot = 0.0
    amp = 1.0
    freq = 1.0
    norm = 0.0
    for i in range(octaves):
        tot += amp * mnoise.noise(v * freq + Vector((i * 13.17, i * 7.31, i * 3.77)))
        norm += amp
        amp *= gain
        freq *= lac
    return tot / norm


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hsv_lin(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b))


# --------------------------------------------------------------------------
# Mesh buffer
# --------------------------------------------------------------------------
class Buf:
    def __init__(self):
        self.verts = []
        self.faces = []
        self.uvs = []

    def add_face(self, idx, uv):
        self.faces.append(tuple(idx))
        self.uvs.append(tuple(uv))

    def to_object(self, name, material=None, smooth=True):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.verts, [], self.faces)
        me.update()
        me.validate(verbose=False)
        if self.uvs and self.faces:
            uvl = me.uv_layers.new(name="UVMap")
            data = uvl.data
            npoly = min(len(me.polygons), len(self.uvs))
            for fi in range(npoly):
                poly = me.polygons[fi]
                uvs = self.uvs[fi]
                for c in range(min(poly.loop_total, len(uvs))):
                    try:
                        data[poly.loop_start + c].uv = uvs[c]
                    except Exception:
                        pass
        ob = bpy.data.objects.new(name, me)
        bpy.context.collection.objects.link(ob)
        if material:
            ob.data.materials.append(material)
        if smooth:
            for p in ob.data.polygons:
                p.use_smooth = True
        return ob


# --------------------------------------------------------------------------
# Tube generation with parallel-transport frames
# --------------------------------------------------------------------------
def frames(path):
    n = len(path)
    T = []
    for i in range(n):
        if i == 0:
            t = path[1] - path[0]
        elif i == n - 1:
            t = path[-1] - path[-2]
        else:
            t = path[i + 1] - path[i - 1]
        if t.length < 1e-9:
            t = Vector((0.0, 0.0, 1.0))
        T.append(t.normalized())

    up = Vector((0.0, 0.0, 1.0))
    if abs(T[0].dot(up)) > 0.9:
        up = Vector((1.0, 0.0, 0.0))
    Nv = [(up - T[0] * T[0].dot(up)).normalized()]
    for i in range(1, n):
        axis = T[i - 1].cross(T[i])
        if axis.length < 1e-9:
            nn = Nv[-1].copy()
        else:
            ang = math.atan2(axis.length, T[i - 1].dot(T[i]))
            q = Quaternion(axis.normalized(), ang)
            nn = q @ Nv[-1]
        nn = nn - T[i] * nn.dot(T[i])
        if nn.length < 1e-9:
            nn = Nv[-1].copy()
        Nv.append(nn.normalized())
    B = [T[i].cross(Nv[i]).normalized() for i in range(n)]
    return T, Nv, B


def add_tube(buf, path, radii, nseg, bark=0.0, seedoff=0.0, cap=False):
    T, N, B = frames(path)
    base = len(buf.verts)
    arc = [0.0]
    for i in range(1, len(path)):
        arc.append(arc[-1] + (path[i] - path[i - 1]).length)

    # UVs are computed PER VERTEX: a per-face u that depends on the local
    # radius would tear the texture horizontally at every ring boundary.
    vuv = []
    for i, p in enumerate(path):
        r = radii[i]
        for j in range(nseg):
            vuv.append(((2.0 * math.pi * j / nseg) * r, arc[i]))

    for i, p in enumerate(path):
        r = radii[i]
        for j in range(nseg):
            a = 2.0 * math.pi * j / nseg
            ca, sa = math.cos(a), math.sin(a)
            rr = r
            if bark > 0.0:
                q = Vector((ca * 2.0, sa * 2.0, arc[i] * 0.30 + seedoff))
                rr = r * (1.0 + bark * (0.62 * fbm(q, 3) + 0.38 * fbm(q * 3.1, 2)))
            buf.verts.append(p + N[i] * (ca * rr) + B[i] * (sa * rr))

    for i in range(len(path) - 1):
        for j in range(nseg):
            j2 = (j + 1) % nseg
            a0 = base + i * nseg + j
            a1 = base + i * nseg + j2
            b0 = base + (i + 1) * nseg + j
            b1 = base + (i + 1) * nseg + j2
            buf.add_face((a0, b0, b1, a1),
                         (vuv[a0 - base], vuv[b0 - base], vuv[b1 - base], vuv[a1 - base]))

    if cap:
        tipc = len(buf.verts)
        buf.verts.append(path[-1])
        last = base + (len(path) - 1) * nseg
        for j in range(nseg):
            j2 = (j + 1) % nseg
            buf.add_face((last + j, tipc, last + j2),
                         (vuv[last + j - base], (2.0 * math.pi * (j + 0.5) / nseg * radii[-1],
                                                 arc[-1] + radii[-1]),
                          vuv[last + j2 - base]))


# --------------------------------------------------------------------------
# Trunk
# --------------------------------------------------------------------------
TRUNK_H = 5.60
R_BASE = 0.465
R_TOP = 0.205
ROOT_COUNT = 7
KNOTS = [(0.55, 0.95), (2.10, 2.10), (3.40, 3.25), (4.85, 1.65), (5.60, 4.35)]


def trunk_radius_at(t):
    r = R_BASE + (R_TOP - R_BASE) * (t ** 0.80)
    r *= (1.0 + 1.05 * math.exp(-t * 24.0))
    r *= (1.0 + 0.045 * math.sin(t * 9.0 + 0.7))
    return r


def trunk_axis(t):
    return Vector((0.115 * math.sin(t * 2.05 + 0.30), 0.085 * math.sin(t * 2.95 + 1.25), t * TRUNK_H))


def build_trunk(buf):
    NH, NR = 200, 128
    base = len(buf.verts)
    # per-vertex UVs: the circumference term depends on the local radius, so
    # it has to be baked per vertex or the texture tears at every ring.
    vuv = []
    for i in range(NH + 1):
        t = i / NH
        z = t * TRUNK_H
        c = trunk_axis(t)
        r = trunk_radius_at(t)
        for j in range(NR):
            vuv.append(((2.0 * math.pi * j / NR) * r, z))
        for j in range(NR):
            a = 2.0 * math.pi * j / NR
            ca, sa = math.cos(a), math.sin(a)
            rr = r

            # buttress roots
            for k in range(ROOT_COUNT):
                ak = 2.0 * math.pi * k / ROOT_COUNT + 0.25
                d = math.atan2(math.sin(a - ak), math.cos(a - ak))
                rr += 0.195 * math.exp(-(d * d) / (2 * 0.135 ** 2)) * math.exp(-t * 8.5) * r

            # burls
            for (ka, kz) in KNOTS:
                da = math.atan2(math.sin(a - ka), math.cos(a - ka))
                dz = z - kz
                rr += 0.080 * math.exp(-(da * da + dz * dz) / (2 * 0.19 ** 2)) * r

            # primary vertical fissure network
            warp = 1.45 * fbm(Vector((ca * 1.35, sa * 1.35, z * 0.20)), 3)
            fis = abs(math.sin((a + warp) * 12.0 + 0.55 * math.sin(z * 0.38)))
            crack = (1.0 - fis) ** 6
            depth = 0.100 * (1.0 + 0.95 * math.exp(-t * 11.0))
            rr -= depth * r * crack

            # secondary finer fissures
            warp2 = 0.9 * fbm(Vector((ca * 3.1, sa * 3.1, z * 0.5)), 2)
            fis2 = abs(math.sin((a + warp2) * 17.0 + 1.7 * math.sin(z * 0.9)))
            rr -= 0.030 * r * (1.0 - fis2) ** 6

            # plate scale only -- the fine grain lives in the shader bump.
            # High-frequency geometry here would alias against the radial grid
            # and show up as a regular moire lattice.
            pl = fbm(Vector((ca * 2.25, sa * 2.25, z * 0.28)), 3)
            rr *= (1.0 + 0.125 * pl)
            fg = fbm(Vector((ca * 4.2, sa * 4.2, z * 1.1)), 2)
            rr *= (1.0 + 0.022 * fg)

            buf.verts.append(c + Vector((ca * rr, sa * rr, 0.0)))

    for i in range(NH):
        for j in range(NR):
            j2 = (j + 1) % NR
            a0 = base + i * NR + j
            a1 = base + i * NR + j2
            b0 = base + (i + 1) * NR + j
            b1 = base + (i + 1) * NR + j2
            buf.add_face((a0, b0, b1, a1),
                         (vuv[a0 - base], vuv[b0 - base], vuv[b1 - base], vuv[a1 - base]))


# --------------------------------------------------------------------------
# Leaf template
# --------------------------------------------------------------------------
LEAF_ROWS = [0.0, 0.14, 0.34, 0.58, 0.80, 1.0]
LEAF_COLS = 5


def leaf_profile(t):
    if t <= 0.0 or t >= 1.0:
        return 0.0
    return max(0.0, math.sin(math.pi * (t ** 0.66)) ** 0.85)


def _seg_dist(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    L2 = vx * vx + vy * vy
    if L2 < 1e-12:
        return math.hypot(wx, wy)
    tt = max(0.0, min(1.0, (wx * vx + wy * vy) / L2))
    return math.hypot(px - (ax + tt * vx), py - (ay + tt * vy))


def make_leaf_template(seed, length, width):
    rng = random.Random(seed)
    vein_ts = [0.20 + 0.16 * k + rng.uniform(-0.02, 0.02) for k in range(4)]
    vein_amp = 0.020 * length * rng.uniform(0.85, 1.2)
    midrib_h = 0.052 * length
    cup = 0.115 * length
    droop = 0.125 * length
    asym = rng.uniform(-0.16, 0.16)

    def relief(x, y, t, u):
        z = -cup * (u * u)
        z += midrib_h * math.exp(-((x / (0.085 * width)) ** 2))
        z -= droop * (t ** 2.1)
        for vt in vein_ts:
            wv = width * 0.5 * leaf_profile(vt)
            for sgn in (-1.0, 1.0):
                ax, ay = 0.0, vt * length
                bx, by = sgn * wv * 0.92, (vt + 0.20) * length
                d = _seg_dist(x, y, ax, ay, bx, by)
                z += vein_amp * math.exp(-((d / (0.055 * length)) ** 2))
        z += 0.010 * length * fbm(Vector((x * 40.0, y * 22.0, 3.3)), 2)
        return z

    rows = []
    for ri, t in enumerate(LEAF_ROWS):
        if ri == 0 or ri == len(LEAF_ROWS) - 1:
            rows.append([(asym * width * 0.10, t * length, relief(0.0, t * length, t, 0.0))])
        else:
            w = width * 0.5 * leaf_profile(t)
            cx = asym * width * 0.22 * math.sin(math.pi * t)
            row = []
            for j in range(LEAF_COLS):
                u = -1.0 + 2.0 * j / (LEAF_COLS - 1)
                x = cx + u * w
                y = t * length
                row.append((x, y, relief(x, y, t, u)))
            rows.append(row)

    verts = []
    idx = []
    for row in rows:
        base = len(verts)
        idx.append(list(range(base, base + len(row))))
        for v in row:
            verts.append(v)

    faces = []
    for j in range(LEAF_COLS - 1):
        faces.append((idx[0][0], idx[1][j], idx[1][j + 1]))
    for ri in range(1, len(rows) - 2):
        A, B = idx[ri], idx[ri + 1]
        for j in range(LEAF_COLS - 1):
            faces.append((A[j], B[j], B[j + 1], A[j + 1]))
    tip = idx[-1][0]
    for j in range(LEAF_COLS - 1):
        faces.append((idx[-2][j], tip, idx[-2][j + 1]))
    return verts, faces


# --------------------------------------------------------------------------
# Foliage placement
# --------------------------------------------------------------------------
class TreeStats:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.n_branches = 0
        self.n_leaves = 0
        self.maxlevel = 4
        self.leaf_cols = []
        self.leaf_rands = []
        self.leaf_counts = []


TEMPLATES = []


def pick_template(rng):
    return TEMPLATES[rng.randrange(len(TEMPLATES))]


def emit_leaf(lbuf, st, pos, forward, up, scale, hsv, rng):
    f = forward + Vector((rng.uniform(-0.45, 0.45), rng.uniform(-0.45, 0.45),
                          rng.uniform(-0.28, 0.20)))
    if f.length < 1e-6:
        f = Vector((0.0, 0.0, 1.0))
    f.normalize()
    side = f.cross(up)
    if side.length < 1e-6:
        side = f.cross(Vector((1.0, 0.0, 0.0)))
    side.normalize()
    nrm = side.cross(f).normalized()
    roll = rng.uniform(-math.pi, math.pi)
    cr, sr = math.cos(roll), math.sin(roll)
    s2 = side * cr + nrm * sr
    n2 = -side * sr + nrm * cr
    m0, m1, m2 = s2.x, s2.y, s2.z
    m3, m4, m5 = f.x, f.y, f.z
    m6, m7, m8 = n2.x, n2.y, n2.z

    tv, tf = pick_template(rng)
    h = hsv[0] + rng.uniform(-0.014, 0.014)
    s = min(1.0, max(0.0, hsv[1] + rng.uniform(-0.11, 0.11)))
    v = min(1.0, max(0.05, hsv[2] * rng.uniform(0.80, 1.20)))
    col = hsv_lin(h, s, v)

    p = pos + Vector((rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), rng.uniform(-0.015, 0.015)))
    px, py, pz = p.x, p.y, p.z
    base = len(lbuf.verts)
    for (vx, vy, vz) in tv:
        lx = vx * scale
        ly = vy * scale
        lz = vz * scale
        lbuf.verts.append((px + m0 * lx + m1 * ly + m2 * lz,
                           py + m3 * lx + m4 * ly + m5 * lz,
                           pz + m6 * lx + m7 * ly + m8 * lz))
    for f_idx in tf:
        lbuf.faces.append(tuple(base + i for i in f_idx))
    lbuf.uvs.append(tuple((0.0, 0.0) for _ in f_idx))
    st.leaf_cols.append(col)
    st.leaf_rands.append(rng.random())
    st.leaf_counts.append(len(tv))
    st.n_leaves += 1


# spacing / start-fraction per branch level (metres along the branch arc)
# levels 1-2 are bare scaffold limbs, foliage lives on the shoots (3-4)
LEAF_RULES = {
    1: (0.30, 0.88),
    2: (0.20, 0.84),
    3: (0.042, 0.08),
    4: (0.042, 0.06),
}
LEAF_PER_SITE = {1: 1, 2: 1, 3: 2, 4: 2}
TIP_CLUSTER = {1: 3, 2: 4, 3: 5, 4: 4}
# how far a leaf may stand off the shoot axis -> breaks the "caterpillar" sheath
SPREAD = {1: 0.05, 2: 0.07, 3: 0.14, 4: 0.13}


def add_branch_leaves(lbuf, st, path, radii, level):
    """Leaves are sprayed SIDEWAYS off the shoot, never stacked along it, so
    the canopy reads as separate fronds instead of a solid green tube."""
    rng = st.rng
    n = len(path)
    spacing, t0 = LEAF_RULES.get(level, LEAF_RULES[max(LEAF_RULES)])
    per = LEAF_PER_SITE.get(level, 1)
    spread = SPREAD.get(level, 0.10)
    arc = [0.0]
    for i in range(1, n):
        arc.append(arc[-1] + (path[i] - path[i - 1]).length)
    total = arc[-1]
    if total < 1e-4:
        return

    i = 0
    while i < n - 1:
        if arc[i] >= t0 * total:
            j = min(n - 2, i + 1)
            p = path[j]
            tv = path[j + 1] - path[j]
            if tv.length < 1e-8:
                i += 1
                continue
            tv.normalize()
            outward = Vector((p.x, p.y, 0.0))
            if outward.length < 1e-5:
                outward = Vector((1.0, 0.0, 0.0))
            outward.normalize()
            side = tv.cross(outward)
            if side.length < 1e-6:
                side = tv.cross(Vector((0.0, 0.0, 1.0)))
            side.normalize()
            core = p + outward * (radii[j] * 0.8)

            for k in range(per):
                ang = rng.uniform(0.0, 2.0 * math.pi)
                # direction perpendicular to the shoot -> leaf fans outward
                perp = (side * math.cos(ang) + outward * math.sin(ang)
                        + Vector((0.0, 0.0, rng.uniform(-0.35, 0.45))))
                perp.normalize()
                anchor = core + perp * (rng.uniform(0.0, spread))
                fwd = (perp * 0.78 + tv * 0.46 + Vector((0.0, 0.0, 0.12))).normalized()
                sc = rng.uniform(0.72, 1.16) * (1.0 + 0.20 * (j / max(1, n - 1)))
                if rng.random() < 0.11:
                    hsv = (rng.uniform(0.085, 0.150), rng.uniform(0.55, 0.92), rng.uniform(0.20, 0.52))
                else:
                    hsv = (rng.uniform(0.200, 0.330), rng.uniform(0.38, 0.90), rng.uniform(0.14, 0.62))
                emit_leaf(lbuf, st, anchor, fwd, Vector((0.0, 0.0, 1.0)), sc, hsv, rng)

            nxt = i
            while nxt < n - 1 and arc[nxt] < arc[i] + spacing:
                nxt += 1
            i = max(nxt, i + 1)
        else:
            i += 1

    # terminal spray
    tip = path[-1]
    t = (path[-1] - path[-2]).normalized() if n > 1 else Vector((0.0, 0.0, 1.0))
    for k in range(TIP_CLUSTER.get(level, 5)):
        ang = rng.uniform(0.0, 2.0 * math.pi)
        rad = Vector((math.cos(ang), math.sin(ang), 0.0))
        fwd = (t * 0.40 + rad * 0.85 + Vector((0.0, 0.0, rng.uniform(-0.05, 0.50)))).normalized()
        sc = rng.uniform(0.68, 1.14)
        if rng.random() < 0.14:
            hsv = (rng.uniform(0.085, 0.150), rng.uniform(0.55, 0.92), rng.uniform(0.20, 0.55))
        else:
            hsv = (rng.uniform(0.200, 0.330), rng.uniform(0.38, 0.90), rng.uniform(0.16, 0.64))
        emit_leaf(lbuf, st, tip + rad * rng.uniform(0.0, 0.06), fwd, Vector((0.0, 0.0, 1.0)),
                  sc, hsv, rng)


# --------------------------------------------------------------------------
# Crown envelope: keeps the canopy a dense rounded dome instead of a spire
# --------------------------------------------------------------------------
CROWN_BASE = 1.90
CROWN_TOP = 7.30
CROWN_R = 3.75
CROWN_APEX = 5.95


def envelope(P, d):
    """Force the canopy into a rounded, multi-lobed crown.  Near the apex
    branches are pushed inward and down, otherwise the leader stacks into a
    spire; the azimuthal lobes keep the dome from reading as a topiary ball."""
    z = P.z
    if z < CROWN_BASE:
        return d
    t = min(1.0, (z - CROWN_BASE) / (CROWN_TOP - CROWN_BASE))
    az = math.atan2(P.y, P.x)
    lobe = (0.80 + 0.26 * math.sin(3.0 * az + 0.65)
            + 0.14 * math.sin(5.0 * az - 1.30))
    allowed = CROWN_R * (max(0.0, 1.0 - t ** 2.4)) ** 0.55 * lobe
    r = math.hypot(P.x, P.y)
    if r > allowed * 0.99:
        inward = Vector((-P.x, -P.y, 0.0))
        if inward.length < 1e-6:
            inward = Vector((1.0, 0.0, 0.0))
        inward.normalize()
        over = (r - allowed) / max(0.40, allowed)
        d = d + inward * min(1.6, over * 2.4)
        if d.z > 0.0:
            d = d - Vector((0.0, 0.0, d.z * min(1.0, over)))
    if z > CROWN_APEX:
        d = d - Vector((0.0, 0.0, max(0.0, d.z) * 1.9 + 0.05))
    return d.normalized()


def grow(buf, lbuf, st, start, direction, length, r0, r1, level):
    rng = st.rng
    st.n_branches += 1
    npts = max(5, int(round(length * 12.0)))
    path = []
    radii = []
    p = start.copy()
    d = direction.normalized()
    droop = 0.075 + 0.055 * level
    seg = length / (npts - 1)
    for i in range(npts):
        t = i / (npts - 1)
        path.append(p.copy())
        radii.append(r0 + (r1 - r0) * (t ** 0.85))
        d = d + Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-0.5, 0.45))) * 0.040
        d = d + Vector((0.0, 0.0, -droop * seg * 0.5))
        d = d.normalized()
        d = envelope(p, d)
        p = p + d * seg

    nseg = 14 if level == 1 else (11 if level == 2 else (9 if level == 3 else (7 if level == 4 else 5)))
    bark = 0.11 if level <= 2 else (0.13 if level == 3 else (0.14 if level == 4 else 0.0))
    add_tube(buf, path, radii, nseg, bark=bark, seedoff=level * 13.7 + rng.uniform(0.0, 40.0),
             cap=(level >= st.maxlevel))

    add_branch_leaves(lbuf, st, path, radii, level)

    if level < st.maxlevel:
        cont = (path[-1] - path[-2]).normalized() if len(path) > 1 else Vector((0.0, 0.0, 1.0))
        cont = (cont + Vector((0.0, 0.0, 0.12))).normalized()
        grow(buf, lbuf, st, path[-1], cont, length * rng.uniform(0.66, 0.80),
             r1 * 0.74, r1 * 0.42, level + 1)

        nlat = 5 if level == 1 else (4 if level == 2 else 4)
        for _ in range(nlat):
            ts = rng.uniform(0.24, 0.82)
            idx = int(ts * (npts - 1))
            idx = max(1, min(npts - 2, idx))
            P = path[idx]
            T = (path[idx + 1] - path[idx - 1]).normalized()
            R = Vector((P.x, P.y, 0.0))
            if R.length < 0.32:
                # near the trunk axis (leader / upper scaffold): fan out randomly
                # instead of shooting straight up, which would give a spire
                a = rng.uniform(0.0, 2.0 * math.pi)
                R = Vector((math.cos(a), math.sin(a), 0.0))
            else:
                R.normalize()
            ang = rng.uniform(-1.0, 1.0) * 1.05
            Tt = Matrix.Rotation(ang, 3, 'Z') @ T
            # shoots (level 3+) spread wider and flatter than scaffold limbs
            if level >= 2:
                cd = (Tt * 0.64 + R * 0.74 + Vector((0.0, 0.0, rng.uniform(-0.10, 0.30)))).normalized()
            else:
                cd = (Tt * 0.60 + R * 0.64 + Vector((0.0, 0.0, 0.44))).normalized()
            grow(buf, lbuf, st, P, cd, length * rng.uniform(0.50, 0.68),
                 radii[idx] * 0.58, radii[idx] * 0.33, level + 1)


def build_tree(buf, lbuf, st):
    rng = st.rng
    build_trunk(buf)

    # dead broken limbs
    for k in range(7):
        t = rng.uniform(0.08, 0.66)
        a = rng.uniform(0.0, 2.0 * math.pi)
        c = trunk_axis(t)
        r = trunk_radius_at(t)
        p0 = c + Vector((math.cos(a) * r * 0.85, math.sin(a) * r * 0.85, 0.0))
        d = (Vector((math.cos(a), math.sin(a), 0.0)) * 0.72 + Vector((0.0, 0.0, 0.72))).normalized()
        L = rng.uniform(0.35, 0.72)
        npts = 7
        path = []
        radii = []
        p = p0.copy()
        dd = d.copy()
        for i in range(npts):
            tt = i / (npts - 1)
            path.append(p.copy())
            radii.append(rng.uniform(0.032, 0.052) * (1.0 - 0.65 * tt))
            dd = (dd + Vector((rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), -0.10))).normalized()
            p = p + dd * (L / (npts - 1))
        add_tube(buf, path, radii, 7, bark=0.16, seedoff=300.0 + k * 9.0, cap=True)

    # primary limbs -- golden angle, broad rounded crown
    n_prim = 20
    golden = math.radians(137.507)
    for k in range(n_prim):
        frac = (k + 0.5) / n_prim
        t = 0.18 + 0.70 * frac + rng.uniform(-0.030, 0.030)
        a = golden * k + rng.uniform(-0.20, 0.20)
        c = trunk_axis(t)
        r = trunk_radius_at(t)
        p0 = c + Vector((math.cos(a) * r * 0.78, math.sin(a) * r * 0.78, 0.0))
        radial = Vector((math.cos(a), math.sin(a), 0.0))
        d = (radial * 0.82 + Vector((0.0, 0.0, 0.60))).normalized()
        length = rng.uniform(1.55, 2.15) * (1.0 - 0.24 * frac)
        r0 = r * rng.uniform(0.26, 0.38)
        grow(buf, lbuf, st, p0, d, length, r0, r0 * 0.56, 1)

    # inner scaffold branches -- these rise through the middle of the crown and
    # close the hole that an outward-radiating limb system leaves behind
    for k in range(13):
        frac = (k + 0.5) / 13.0
        t = 0.28 + 0.62 * frac
        a = golden * (k + 0.5) + rng.uniform(-0.3, 0.3)
        c = trunk_axis(t)
        r = trunk_radius_at(t)
        p0 = c + Vector((math.cos(a) * r * 0.72, math.sin(a) * r * 0.72, 0.0))
        radial = Vector((math.cos(a), math.sin(a), 0.0))
        d = (radial * 0.58 + Vector((0.0, 0.0, 0.82))).normalized()
        length = rng.uniform(1.15, 1.85) * (1.0 - 0.30 * frac)
        r0 = r * rng.uniform(0.14, 0.22)
        grow(buf, lbuf, st, p0, d, length, r0, r0 * 0.56, 2)

    # crown apex: several short branches of similar height so the top of the
    # tree becomes a cluster of rounded peaks instead of one spire
    for k in range(5):
        a = golden * (k + 0.9) + rng.uniform(-0.25, 0.25)
        c = trunk_axis(0.99)
        r = trunk_radius_at(0.99)
        p0 = c + Vector((math.cos(a) * r * 0.55, math.sin(a) * r * 0.55, -0.18))
        radial = Vector((math.cos(a), math.sin(a), 0.0))
        d = (radial * rng.uniform(0.42, 0.70) + Vector((0.0, 0.0, 0.78))).normalized()
        length = rng.uniform(0.85, 1.25)
        r0 = r * rng.uniform(0.11, 0.17)
        grow(buf, lbuf, st, p0, d, length, r0, r0 * 0.54, 2)

    # leader -- short, so the crown stays rounded instead of spired
    c = trunk_axis(1.0)
    r = trunk_radius_at(1.0)
    p0 = c + Vector((0.0, 0.0, -0.30))
    d = Vector((0.38, -0.24, 0.89)).normalized()
    grow(buf, lbuf, st, p0, d, 1.15, r * 0.55, r * 0.32, 1)


# --------------------------------------------------------------------------
# Ground + litter
# --------------------------------------------------------------------------
GROUND_SIZE = 420.0
GROUND_SEGS = 300
MOUND_R = 2.45


def ground_h(x, y):
    r = math.hypot(x, y)
    h = 0.55 * math.exp(-(r / MOUND_R) ** 2)
    h += 3.20 * fbm(Vector((x * 0.020, y * 0.020, 2.7)), 4)
    h += 1.05 * fbm(Vector((x * 0.075, y * 0.075, 5.3)), 4)
    h += 0.30 * fbm(Vector((x * 0.31, y * 0.31, 8.1)), 3)
    h += 0.085 * fbm(Vector((x * 0.85, y * 0.85, 4.4)), 3)
    # flatten the immediate surroundings of the trunk a little
    h *= 0.50 + 0.50 * min(1.0, r / 1.6)
    return h


def build_ground(buf):
    S = GROUND_SEGS
    W = S + 1
    for i in range(W):
        for j in range(W):
            x = (i / S - 0.5) * GROUND_SIZE
            y = (j / S - 0.5) * GROUND_SIZE
            buf.verts.append((x, y, ground_h(x, y)))
    for i in range(S):
        for j in range(S):
            a = i * W + j
            b = (i + 1) * W + j
            c = (i + 1) * W + j + 1
            d = i * W + j + 1
            buf.add_face((a, b, c, d), ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)))


def build_grass(buf, st, count=220000, radius=15.5, tall=False, clump_thr=-0.20):
    """Individual grass blades -- a forest floor needs undergrowth, otherwise
    the ground reads as desert sand.  Density is modulated by a low frequency
    noise so the meadow breaks into natural patches of bare earth instead of
    ending at a hard circular boundary."""
    rng = st.rng
    SEG = 2
    for _ in range(count):
        a = rng.uniform(0.0, 2.0 * math.pi)
        r = radius * (rng.random() ** 0.5)
        x, y = math.cos(a) * r, math.sin(a) * r
        clump = fbm(Vector((x * 0.075, y * 0.075, 21.7)), 3)
        if clump < clump_thr:
            continue
        z = ground_h(x, y) - 0.012
        if tall:
            h = rng.uniform(0.18, 0.46)
            w = rng.uniform(0.004, 0.009)
        else:
            h = rng.uniform(0.07, 0.21) * (1.0 - 0.35 * min(1.0, r / radius))
            w = rng.uniform(0.0045, 0.0095)
        lean = rng.uniform(0.15, 0.95)
        dang = rng.uniform(0.0, 2.0 * math.pi)
        dx, dy = math.cos(dang), math.sin(dang)
        sx, sy = -dy, dx
        if rng.random() < 0.93:
            col = hsv_lin(rng.uniform(0.215, 0.305), rng.uniform(0.40, 0.88), rng.uniform(0.068, 0.30))
        else:
            col = hsv_lin(rng.uniform(0.090, 0.160), rng.uniform(0.22, 0.52), rng.uniform(0.035, 0.13))
        base = len(buf.verts)
        for i in range(SEG + 1):
            t = i / SEG
            bend = h * lean * (t ** 1.8)
            up = h * (t - 0.34 * t * t)
            ww = w * ((1.0 - t) ** 0.65)
            cx = x + dx * bend
            cy = y + dy * bend
            cz = z + up
            f = 0.38 + 0.62 * t
            c0 = (col[0] * f, col[1] * f, col[2] * f)
            buf.verts.append((cx - sx * ww, cy - sy * ww, cz))
            buf.verts.append((cx + sx * ww, cy + sy * ww, cz))
            for _ in range(2):
                st.leaf_cols.append(c0)
                st.leaf_rands.append(rng.random())
                st.leaf_counts.append(1)
            st.n_leaves += 1
        for i in range(SEG):
            a0 = base + i * 2
            buf.add_face((a0, a0 + 2, a0 + 3, a0 + 1),
                         ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)))


def build_litter(buf, st, count=2600):
    rng = st.rng
    for _ in range(count):
        a = rng.uniform(0.0, 2.0 * math.pi)
        r = 0.5 + (rng.random() ** 0.45) * 16.0
        x, y = math.cos(a) * r, math.sin(a) * r
        z = ground_h(x, y) + 0.004
        tv, tf = pick_template(rng)
        ang = rng.uniform(0.0, 2.0 * math.pi)
        sc = rng.uniform(0.52, 0.82)
        fx, fy = math.cos(ang), math.sin(ang)
        fz = rng.uniform(-0.10, 0.10)
        fl = math.sqrt(fx * fx + fy * fy + fz * fz)
        fx, fy, fz = fx / fl, fy / fl, fz / fl
        sx, sy = -fy, fx
        nz = 1.0
        cr, sr = math.cos(rng.uniform(-0.3, 0.3)), math.sin(rng.uniform(-0.3, 0.3))
        s2x, s2y = sx * cr, sy * cr
        s2z = nz * sr
        n2x, n2y, n2z = -sx * sr, -sy * sr, nz * cr
        m0, m1, m2 = s2x, s2y, s2z
        m3, m4, m5 = fx, fy, fz
        m6, m7, m8 = n2x, n2y, n2z
        col = hsv_lin(rng.uniform(0.030, 0.115), rng.uniform(0.40, 0.80), rng.uniform(0.07, 0.26))
        base = len(buf.verts)
        for (vx, vy, vz) in tv:
            lx, ly, lz = vx * sc, vy * sc, vz * sc
            buf.verts.append((x + m0 * lx + m1 * ly + m2 * lz,
                              y + m3 * lx + m4 * ly + m5 * lz,
                              z + m6 * lx + m7 * ly + m8 * lz))
        for f_idx in tf:
            buf.faces.append(tuple(base + i for i in f_idx))
        buf.uvs.append(tuple((0.0, 0.0) for _ in f_idx))
        st.leaf_cols.append(col)
        st.leaf_rands.append(rng.random())
        st.leaf_counts.append(len(tv))


# --------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------
def new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (900, 0)
    return m, nt, out


def make_bark_material():
    m, nt, out = new_mat("Bark")
    L = nt.links
    tc = nt.nodes.new("ShaderNodeTexCoord"); tc.location = (-1400, 0)
    mp = nt.nodes.new("ShaderNodeMapping"); mp.location = (-1200, 0)
    mp.inputs["Scale"].default_value = (1.9, 0.48, 1.0)
    L.new(tc.outputs["UV"], mp.inputs["Vector"])

    # --- domain warp -----------------------------------------------------
    # Two scales: a high-frequency scramble that breaks the Voronoi lattice
    # cell by cell, plus a low-frequency waviness for organic drift.
    wn = nt.nodes.new("ShaderNodeTexNoise"); wn.location = (-1400, -520)
    wn.inputs["Scale"].default_value = 6.5
    wn.inputs["Detail"].default_value = 2.0
    wn.inputs["Roughness"].default_value = 0.55
    L.new(mp.outputs["Vector"], wn.inputs["Vector"])
    wn2 = nt.nodes.new("ShaderNodeTexNoise"); wn2.location = (-1400, -760)
    wn2.inputs["Scale"].default_value = 1.5
    wn2.inputs["Detail"].default_value = 3.0
    L.new(mp.outputs["Vector"], wn2.inputs["Vector"])

    def _signed(node, amount, loc):
        s = nt.nodes.new("ShaderNodeVectorMath"); s.location = loc
        s.operation = "SUBTRACT"
        s.inputs[1].default_value = (0.5, 0.5, 0.5)
        L.new(node.outputs["Color"], s.inputs[0])
        sc = nt.nodes.new("ShaderNodeVectorMath"); sc.location = (loc[0] + 160, loc[1])
        sc.operation = "SCALE"
        sc.inputs["Scale"].default_value = amount
        L.new(s.outputs["Vector"], sc.inputs[0])
        return sc

    wa = _signed(wn, 1.15, (-1200, -520))
    wb = _signed(wn2, 0.85, (-1200, -760))
    wsum = nt.nodes.new("ShaderNodeVectorMath"); wsum.location = (-880, -620)
    wsum.operation = "ADD"
    L.new(wa.outputs["Vector"], wsum.inputs[0])
    L.new(wb.outputs["Vector"], wsum.inputs[1])
    # keep the warp strictly 2D: adding a fast varying Z would slice the 3D
    # noise / voronoi cells into blocky fragments
    w2d = nt.nodes.new("ShaderNodeVectorMath"); w2d.location = (-760, -620)
    w2d.operation = "MULTIPLY"
    w2d.inputs[1].default_value = (1.0, 1.0, 0.0)
    L.new(wsum.outputs["Vector"], w2d.inputs[0])
    wadd = nt.nodes.new("ShaderNodeVectorMath"); wadd.location = (-1010, -260)
    wadd.operation = "ADD"
    L.new(mp.outputs["Vector"], wadd.inputs[0])
    L.new(w2d.outputs["Vector"], wadd.inputs[1])
    warped = wadd.outputs["Vector"]

    n1 = nt.nodes.new("ShaderNodeTexNoise"); n1.location = (-980, 260)
    n1.inputs["Scale"].default_value = 3.2
    n1.inputs["Detail"].default_value = 2.0
    n1.inputs["Roughness"].default_value = 0.5
    n1.inputs["Distortion"].default_value = 0.15
    L.new(warped, n1.inputs["Vector"])

    vor = nt.nodes.new("ShaderNodeTexVoronoi"); vor.location = (-980, -40)
    vor.voronoi_dimensions = "2D"
    vor.feature = "DISTANCE_TO_EDGE"
    vor.inputs["Scale"].default_value = 5.0
    vor.inputs["Randomness"].default_value = 1.0
    L.new(warped, vor.inputs["Vector"])

    crack_ramp = nt.nodes.new("ShaderNodeValToRGB"); crack_ramp.location = (-780, -40)
    e = crack_ramp.color_ramp.elements
    e[0].position = 0.0
    e[0].color = (1.0, 1.0, 1.0, 1.0)
    e[1].position = 0.055
    e[1].color = (0.0, 0.0, 0.0, 1.0)
    L.new(vor.outputs["Distance"], crack_ramp.inputs["Factor"])

    col_ramp = nt.nodes.new("ShaderNodeValToRGB"); col_ramp.location = (-780, 300)
    e = col_ramp.color_ramp.elements
    e[0].position = 0.28
    e[0].color = (0.0035, 0.0026, 0.0016, 1.0)
    e[1].position = 0.74
    e[1].color = (0.0225, 0.0150, 0.0082, 1.0)
    mid = col_ramp.color_ramp.elements.new(0.52)
    mid.color = (0.100, 0.068, 0.041, 1.0)
    L.new(n1.outputs["Factor"], col_ramp.inputs["Factor"])

    mix1 = nt.nodes.new("ShaderNodeMixRGB"); mix1.location = (-540, 220)
    mix1.inputs["Color2"].default_value = (0.014, 0.009, 0.005, 1.0)
    L.new(crack_ramp.outputs["Color"], mix1.inputs["Factor"])
    L.new(col_ramp.outputs["Color"], mix1.inputs["Color1"])

    lich = nt.nodes.new("ShaderNodeTexNoise"); lich.location = (-980, -320)
    lich.inputs["Scale"].default_value = 1.6
    lich.inputs["Detail"].default_value = 3.0
    L.new(mp.outputs["Vector"], lich.inputs["Vector"])
    lich_ramp = nt.nodes.new("ShaderNodeValToRGB"); lich_ramp.location = (-780, -320)
    e = lich_ramp.color_ramp.elements
    e[0].position = 0.62
    e[1].position = 0.80
    e[0].color = (0.0, 0.0, 0.0, 1.0)
    e[1].color = (1.0, 1.0, 1.0, 1.0)
    L.new(lich.outputs["Factor"], lich_ramp.inputs["Factor"])
    mix2 = nt.nodes.new("ShaderNodeMixRGB"); mix2.location = (-340, 160)
    mix2.inputs["Color2"].default_value = (0.145, 0.142, 0.114, 1.0)
    L.new(lich_ramp.outputs["Color"], mix2.inputs["Factor"])
    L.new(mix1.outputs["Color"], mix2.inputs["Color1"])

    geo = nt.nodes.new("ShaderNodeNewGeometry"); geo.location = (-1400, -520)
    sep = nt.nodes.new("ShaderNodeSeparateXYZ"); sep.location = (-1220, -520)
    L.new(geo.outputs["Position"], sep.inputs["Vector"])
    mr = nt.nodes.new("ShaderNodeMapRange"); mr.location = (-1040, -520)
    mr.inputs["From Min"].default_value = 0.10
    mr.inputs["From Max"].default_value = 2.10
    mr.inputs["To Min"].default_value = 1.0
    mr.inputs["To Max"].default_value = 0.0
    L.new(sep.outputs["Z"], mr.inputs["Value"])
    moss_n = nt.nodes.new("ShaderNodeTexNoise"); moss_n.location = (-1040, -700)
    moss_n.inputs["Scale"].default_value = 5.0
    moss_n.inputs["Detail"].default_value = 3.0
    L.new(tc.outputs["Object"], moss_n.inputs["Vector"])
    moss_r = nt.nodes.new("ShaderNodeValToRGB"); moss_r.location = (-860, -700)
    e = moss_r.color_ramp.elements
    e[0].position = 0.48
    e[1].position = 0.72
    L.new(moss_n.outputs["Factor"], moss_r.inputs["Factor"])
    mul = nt.nodes.new("ShaderNodeMath"); mul.location = (-680, -600)
    mul.operation = "MULTIPLY"
    L.new(mr.outputs["Result"], mul.inputs[0])
    L.new(moss_r.outputs["Color"], mul.inputs[1])
    mix3 = nt.nodes.new("ShaderNodeMixRGB"); mix3.location = (-140, 100)
    mix3.inputs["Color2"].default_value = (0.040, 0.078, 0.026, 1.0)
    L.new(mul.outputs["Value"], mix3.inputs["Factor"])
    L.new(mix2.outputs["Color"], mix3.inputs["Color1"])

    fine = nt.nodes.new("ShaderNodeTexNoise"); fine.location = (-980, -140)
    fine.inputs["Scale"].default_value = 8.0
    fine.inputs["Detail"].default_value = 2.0
    fine.inputs["Roughness"].default_value = 0.6
    L.new(warped, fine.inputs["Vector"])

    # second, finer crack network for the small surface checks
    vor2 = nt.nodes.new("ShaderNodeTexVoronoi"); vor2.location = (-980, -300)
    vor2.voronoi_dimensions = "2D"
    vor2.feature = "DISTANCE_TO_EDGE"
    vor2.inputs["Scale"].default_value = 4.5
    vor2.inputs["Randomness"].default_value = 1.0
    L.new(warped, vor2.inputs["Vector"])
    crack2 = nt.nodes.new("ShaderNodeValToRGB"); crack2.location = (-780, -300)
    e = crack2.color_ramp.elements
    e[0].position = 0.0
    e[0].color = (1.0, 1.0, 1.0, 1.0)
    e[1].position = 0.10
    e[1].color = (0.0, 0.0, 0.0, 1.0)
    L.new(vor2.outputs["Distance"], crack2.inputs["Factor"])

    hmix = nt.nodes.new("ShaderNodeMixRGB"); hmix.location = (-560, -200)
    hmix.inputs["Factor"].default_value = 0.42
    L.new(crack_ramp.outputs["Color"], hmix.inputs["Color1"])
    L.new(crack2.outputs["Color"], hmix.inputs["Color2"])
    # invert: bark PLATES stand proud, the crack network sinks into grooves
    inv = nt.nodes.new("ShaderNodeMixRGB"); inv.location = (-640, -60)
    inv.inputs["Color1"].default_value = (1.0, 1.0, 1.0, 1.0)
    inv.inputs["Color2"].default_value = (0.0, 0.0, 0.0, 1.0)
    L.new(hmix.outputs["Color"], inv.inputs["Factor"])
    hmix2 = nt.nodes.new("ShaderNodeMixRGB"); hmix2.location = (-460, -200)
    hmix2.inputs["Factor"].default_value = 0.26
    L.new(inv.outputs["Color"], hmix2.inputs["Color1"])
    L.new(fine.outputs["Fac"], hmix2.inputs["Color2"])
    bump = nt.nodes.new("ShaderNodeBump"); bump.location = (-340, -220)
    bump.inputs["Strength"].default_value = 0.90
    bump.inputs["Distance"].default_value = 0.030
    # a wide filter keeps the screen-space derivative from aliasing into
    # visible blocky stair-steps on the trunk
    bump.inputs["Filter Width"].default_value = 1.7
    L.new(hmix2.outputs["Color"], bump.inputs["Height"])

    rough = nt.nodes.new("ShaderNodeMapRange"); rough.location = (-340, -400)
    rough.inputs["To Min"].default_value = 0.95
    rough.inputs["To Max"].default_value = 0.72
    L.new(n1.outputs["Factor"], rough.inputs["Value"])

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (250, 0)
    L.new(mix3.outputs["Color"], bsdf.inputs["Base Color"])
    L.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    # low specular: a grazing sheen on the trunk is what made the bark read
    # as polished marble instead of rough bark
    bsdf.inputs["Specular IOR Level"].default_value = 0.08
    L.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return m


def make_foliage_material(translucent=True):
    m, nt, out = new_mat("Foliage" if translucent else "Litter")
    L = nt.links
    rough = 0.34 if translucent else 0.82
    spec = 0.45 if translucent else 0.12
    attr = nt.nodes.new("ShaderNodeAttribute"); attr.location = (-900, 200)
    attr.attribute_type = "GEOMETRY"
    attr.attribute_name = "Col"

    tc = nt.nodes.new("ShaderNodeTexCoord"); tc.location = (-1100, -200)
    speck = nt.nodes.new("ShaderNodeTexNoise"); speck.location = (-900, -220)
    speck.inputs["Scale"].default_value = 70.0
    speck.inputs["Detail"].default_value = 2.0
    L.new(tc.outputs["Object"], speck.inputs["Vector"])

    hsv = nt.nodes.new("ShaderNodeHueSaturation"); hsv.location = (-640, 200)
    hsv.inputs["Saturation"].default_value = 1.10
    L.new(attr.outputs["Color"], hsv.inputs["Color"])

    mixc = nt.nodes.new("ShaderNodeMixRGB"); mixc.location = (-430, 200)
    mixc.blend_type = "MULTIPLY"
    mixc.inputs["Factor"].default_value = 0.16
    L.new(hsv.outputs["Color"], mixc.inputs["Color1"])
    L.new(speck.outputs["Color"], mixc.inputs["Color2"])

    bump = nt.nodes.new("ShaderNodeBump"); bump.location = (-430, -260)
    bump.inputs["Strength"].default_value = 0.16
    bump.inputs["Distance"].default_value = 0.005
    L.new(speck.outputs["Fac"], bump.inputs["Height"])

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (-120, 120)
    L.new(mixc.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["IOR"].default_value = 1.45
    bsdf.inputs["Specular IOR Level"].default_value = spec
    bsdf.inputs["Sheen Weight"].default_value = 0.14 if translucent else 0.0
    bsdf.inputs["Sheen Roughness"].default_value = 0.35
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    if not translucent:
        L.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        return m

    trans = nt.nodes.new("ShaderNodeBsdfTranslucent"); trans.location = (-120, -180)
    tb = nt.nodes.new("ShaderNodeHueSaturation"); tb.location = (-330, -380)
    tb.inputs["Value"].default_value = 3.00
    tb.inputs["Saturation"].default_value = 0.95
    L.new(attr.outputs["Color"], tb.inputs["Color"])
    L.new(tb.outputs["Color"], trans.inputs["Color"])

    mixsh = nt.nodes.new("ShaderNodeMixShader"); mixsh.location = (300, 0)
    mixsh.inputs["Fac"].default_value = 0.68
    L.new(bsdf.outputs["BSDF"], mixsh.inputs[1])
    L.new(trans.outputs["BSDF"], mixsh.inputs[2])
    L.new(mixsh.outputs["Shader"], out.inputs["Surface"])
    return m


def make_grass_material():
    """Backlit grass: strong translucency so the low sun ignites the blades."""
    m, nt, out = new_mat("Grass")
    L = nt.links
    attr = nt.nodes.new("ShaderNodeAttribute"); attr.location = (-900, 200)
    attr.attribute_type = "GEOMETRY"
    attr.attribute_name = "Col"

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (-200, 120)
    L.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.64
    bsdf.inputs["Specular IOR Level"].default_value = 0.10

    trans = nt.nodes.new("ShaderNodeBsdfTranslucent"); trans.location = (-200, -200)
    tb = nt.nodes.new("ShaderNodeHueSaturation"); tb.location = (-450, -380)
    tb.inputs["Value"].default_value = 1.18
    tb.inputs["Saturation"].default_value = 0.95
    L.new(attr.outputs["Color"], tb.inputs["Color"])
    L.new(tb.outputs["Color"], trans.inputs["Color"])

    mixsh = nt.nodes.new("ShaderNodeMixShader"); mixsh.location = (250, 0)
    mixsh.inputs["Fac"].default_value = 0.50
    L.new(bsdf.outputs["BSDF"], mixsh.inputs[1])
    L.new(trans.outputs["BSDF"], mixsh.inputs[2])
    L.new(mixsh.outputs["Shader"], out.inputs["Surface"])
    return m


def make_ground_material():
    """Dark forest floor: damp loam, moss patches, dry grass litter."""
    m, nt, out = new_mat("ForestFloor")
    L = nt.links
    tc = nt.nodes.new("ShaderNodeTexCoord"); tc.location = (-1200, 0)

    n1 = nt.nodes.new("ShaderNodeTexNoise"); n1.location = (-1000, 200)
    n1.inputs["Scale"].default_value = 0.55
    n1.inputs["Detail"].default_value = 4.0
    n1.inputs["Roughness"].default_value = 0.70
    L.new(tc.outputs["Object"], n1.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB"); ramp.location = (-800, 200)
    e = ramp.color_ramp.elements
    e[0].position = 0.30
    e[0].color = (0.0050, 0.0040, 0.0026, 1.0)
    e[1].position = 0.74
    e[1].color = (0.022, 0.0155, 0.0085, 1.0)
    mid = ramp.color_ramp.elements.new(0.55)
    mid.color = (0.010, 0.0072, 0.0040, 1.0)
    L.new(n1.outputs["Factor"], ramp.inputs["Factor"])

    # moss
    n2 = nt.nodes.new("ShaderNodeTexNoise"); n2.location = (-1000, -100)
    n2.inputs["Scale"].default_value = 1.9
    n2.inputs["Detail"].default_value = 3.0
    n2.inputs["Roughness"].default_value = 0.75
    L.new(tc.outputs["Object"], n2.inputs["Vector"])
    moss = nt.nodes.new("ShaderNodeValToRGB"); moss.location = (-800, -100)
    e = moss.color_ramp.elements
    e[0].position = 0.44
    e[1].position = 0.60
    L.new(n2.outputs["Factor"], moss.inputs["Factor"])
    mix = nt.nodes.new("ShaderNodeMixRGB"); mix.location = (-560, 120)
    mix.inputs["Color2"].default_value = (0.011, 0.026, 0.0065, 1.0)
    L.new(moss.outputs["Color"], mix.inputs["Factor"])
    L.new(ramp.outputs["Color"], mix.inputs["Color1"])

    # dry grass / straw litter
    n3 = nt.nodes.new("ShaderNodeTexNoise"); n3.location = (-1000, -400)
    n3.inputs["Scale"].default_value = 6.5
    n3.inputs["Detail"].default_value = 3.0
    L.new(tc.outputs["Object"], n3.inputs["Vector"])
    straw = nt.nodes.new("ShaderNodeValToRGB"); straw.location = (-800, -400)
    e = straw.color_ramp.elements
    e[0].position = 0.70
    e[1].position = 0.84
    L.new(n3.outputs["Factor"], straw.inputs["Factor"])
    mix2 = nt.nodes.new("ShaderNodeMixRGB"); mix2.location = (-360, 60)
    mix2.inputs["Color2"].default_value = (0.019, 0.0135, 0.0058, 1.0)
    L.new(straw.outputs["Color"], mix2.inputs["Factor"])
    L.new(mix.outputs["Color"], mix2.inputs["Color1"])

    fine = nt.nodes.new("ShaderNodeTexNoise"); fine.location = (-1000, -660)
    fine.inputs["Scale"].default_value = 22.0
    fine.inputs["Detail"].default_value = 2.0
    L.new(tc.outputs["Object"], fine.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump"); bump.location = (-360, -380)
    bump.inputs["Strength"].default_value = 0.6
    bump.inputs["Distance"].default_value = 0.03
    L.new(fine.outputs["Fac"], bump.inputs["Height"])

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (-80, 0)
    L.new(mix2.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.97
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    # A 420 m plane seen from 0.95 m is almost entirely at grazing incidence,
    # where even a rough dielectric mirrors the sky at near 100 % and turns
    # the forest floor into bright sand.  Soil is Lambertian -- kill specular.
    bsdf.inputs["Specular IOR Level"].default_value = 0.0
    L.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return m


# --------------------------------------------------------------------------
# World, lights, cameras
# --------------------------------------------------------------------------
def setup_world(sun_az, sun_el):
    """Physically-based sky + procedural cloud deck + warm horizon band."""
    w = bpy.data.worlds.new("Sky")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld"); out.location = (900, 0)
    bg = nt.nodes.new("ShaderNodeBackground"); bg.location = (700, 0)
    bg.inputs["Strength"].default_value = 0.135

    sky = nt.nodes.new("ShaderNodeTexSky"); sky.location = (-500, 250)
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_disc = True
    sky.sun_size = math.radians(0.62)
    sky.sun_intensity = 0.40
    sky.sun_elevation = sun_el
    sky.sun_rotation = sun_az
    sky.altitude = 120.0
    sky.air_density = 1.05
    sky.aerosol_density = 1.7
    sky.ozone_density = 1.0

    # --- cloud deck ------------------------------------------------------
    tcw = nt.nodes.new("ShaderNodeTexCoord"); tcw.location = (-900, -180)
    cwarp = nt.nodes.new("ShaderNodeVectorMath"); cwarp.location = (-700, -180)
    cwarp.operation = "MULTIPLY"
    cwarp.inputs[1].default_value = (2.1, 2.1, 7.5)
    L_new = nt.links.new
    L_new(tcw.outputs["Generated"], cwarp.inputs[0])

    cnoise = nt.nodes.new("ShaderNodeTexNoise"); cnoise.location = (-500, -180)
    cnoise.inputs["Scale"].default_value = 2.0
    cnoise.inputs["Detail"].default_value = 7.0
    cnoise.inputs["Roughness"].default_value = 0.55
    L_new(cwarp.outputs["Vector"], cnoise.inputs["Vector"])
    cramp = nt.nodes.new("ShaderNodeValToRGB"); cramp.location = (-300, -180)
    e = cramp.color_ramp.elements
    e[0].position = 0.46
    e[0].color = (0.0, 0.0, 0.0, 1.0)
    e[1].position = 0.68
    e[1].color = (1.0, 1.0, 1.0, 1.0)
    L_new(cnoise.outputs["Factor"], cramp.inputs["Factor"])

    # clouds catch the low sun and burn warm on that side
    sundir = Vector((math.cos(sun_az) * math.cos(sun_el),
                     math.sin(sun_az) * math.cos(sun_el),
                     math.sin(sun_el)))
    dot = nt.nodes.new("ShaderNodeVectorMath"); dot.location = (-500, -420)
    dot.operation = "DOT_PRODUCT"
    dot.inputs[1].default_value = sundir
    L_new(tcw.outputs["Generated"], dot.inputs[0])
    dmap = nt.nodes.new("ShaderNodeMapRange"); dmap.location = (-300, -420)
    dmap.inputs["From Min"].default_value = 0.2
    dmap.inputs["From Max"].default_value = 1.0
    dmap.inputs["To Min"].default_value = 0.72
    dmap.inputs["To Max"].default_value = 1.25
    L_new(dot.outputs["Value"], dmap.inputs["Value"])

    cc = nt.nodes.new("ShaderNodeCombineXYZ"); cc.location = (-120, -420)
    L_new(dmap.outputs["Result"], cc.inputs["X"])
    L_new(dmap.outputs["Result"], cc.inputs["Y"])
    L_new(dmap.outputs["Result"], cc.inputs["Z"])
    ccol = nt.nodes.new("ShaderNodeRGB"); ccol.location = (-300, -600)
    ccol.outputs[0].default_value = (1.0, 0.88, 0.74, 1.0)

    cmix = nt.nodes.new("ShaderNodeMixRGB"); cmix.location = (100, 100)
    cmix.inputs["Color2"].default_value = (1.0, 0.90, 0.78, 1.0)
    L_new(cramp.outputs["Color"], cmix.inputs["Factor"])
    L_new(sky.outputs["Color"], cmix.inputs["Color1"])
    ctint = nt.nodes.new("ShaderNodeMixRGB"); ctint.location = (300, 100)
    ctint.blend_type = "MULTIPLY"
    ctint.inputs["Factor"].default_value = 0.55
    L_new(cmix.outputs["Color"], ctint.inputs["Color1"])
    L_new(cc.outputs["Vector"], ctint.inputs["Color2"])

    # --- warm horizon band ----------------------------------------------
    sep = nt.nodes.new("ShaderNodeSeparateXYZ"); sep.location = (-700, -700)
    L_new(tcw.outputs["Generated"], sep.inputs["Vector"])
    hz = nt.nodes.new("ShaderNodeMapRange"); hz.location = (-500, -700)
    hz.inputs["From Min"].default_value = -0.02
    hz.inputs["From Max"].default_value = 0.26
    hz.inputs["To Min"].default_value = 1.0
    hz.inputs["To Max"].default_value = 0.0
    L_new(sep.outputs["Z"], hz.inputs["Value"])
    hzf = nt.nodes.new("ShaderNodeMath"); hzf.location = (-300, -700)
    hzf.operation = "MULTIPLY"
    hzf.inputs[1].default_value = 0.50
    L_new(hz.outputs["Result"], hzf.inputs[0])
    hcol = nt.nodes.new("ShaderNodeRGB"); hcol.location = (-300, -880)
    hcol.outputs[0].default_value = (1.0, 0.66, 0.38, 1.0)
    hmix = nt.nodes.new("ShaderNodeMixRGB"); hmix.location = (500, 100)
    L_new(hzf.outputs["Value"], hmix.inputs["Factor"])
    L_new(ctint.outputs["Color"], hmix.inputs["Color1"])
    L_new(hcol.outputs[0], hmix.inputs["Color2"])

    L_new(hmix.outputs["Color"], bg.inputs["Color"])
    L_new(bg.outputs["Background"], out.inputs["Surface"])
    return w


def add_sun(sun_az, sun_el):
    d = Vector((math.cos(sun_az) * math.cos(sun_el),
                math.sin(sun_az) * math.cos(sun_el),
                math.sin(sun_el)))
    ld = bpy.data.lights.new("Sun", type="SUN")
    ld.energy = 12.5
    ld.color = (1.0, 0.700, 0.395)
    ld.angle = math.radians(1.0)
    ob = bpy.data.objects.new("Sun", ld)
    bpy.context.collection.objects.link(ob)
    ob.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    ob.location = d * 60.0
    return ob


def add_fill(cam_az):
    az = cam_az + math.radians(62.0)
    el = math.radians(40.0)
    d = Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)))
    ld = bpy.data.lights.new("Fill", type="AREA")
    ld.shape = "RECTANGLE"
    ld.size = 22.0
    ld.size_y = 14.0
    ld.energy = 1700.0
    ld.color = (0.52, 0.68, 1.0)
    ob = bpy.data.objects.new("Fill", ld)
    bpy.context.collection.objects.link(ob)
    ob.location = d * 22.0
    ob.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    return ob


def add_bounce():
    ld = bpy.data.lights.new("Bounce", type="AREA")
    ld.shape = "RECTANGLE"
    ld.size = 22.0
    ld.size_y = 22.0
    ld.energy = 700.0
    ld.color = (0.90, 0.76, 0.54)
    ob = bpy.data.objects.new("Bounce", ld)
    bpy.context.collection.objects.link(ob)
    ob.location = Vector((0.0, 0.0, 0.10))
    ob.rotation_euler = (math.pi, 0.0, 0.0)
    ob.visible_shadow = False
    return ob


def add_mist():
    """Low ground mist: density falls off with height so it hugs the soil
    instead of veiling the whole frame."""
    m = bpy.data.materials.new("Mist")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (400, 0)
    sc = nt.nodes.new("ShaderNodeVolumeScatter"); sc.location = (150, 0)
    sc.inputs["Color"].default_value = (0.74, 0.82, 0.96, 1.0)
    sc.inputs["Anisotropy"].default_value = 0.28
    geo = nt.nodes.new("ShaderNodeNewGeometry"); geo.location = (-400, -220)
    sep = nt.nodes.new("ShaderNodeSeparateXYZ"); sep.location = (-220, -220)
    nt.links.new(geo.outputs["Position"], sep.inputs["Vector"])
    hmap = nt.nodes.new("ShaderNodeMapRange"); hmap.location = (-40, -220)
    hmap.inputs["From Min"].default_value = 0.30
    hmap.inputs["From Max"].default_value = 3.40
    hmap.inputs["To Min"].default_value = 1.0
    hmap.inputs["To Max"].default_value = 0.0
    nt.links.new(sep.outputs["Z"], hmap.inputs["Value"])
    mul = nt.nodes.new("ShaderNodeMath"); mul.location = (0, 0)
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = MIST_DENSITY
    nt.links.new(hmap.outputs["Result"], mul.inputs[0])
    nt.links.new(mul.outputs["Value"], sc.inputs["Density"])
    nt.links.new(sc.outputs["Volume"], out.inputs["Volume"])
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 8))
    ob = bpy.context.active_object
    ob.name = "MistVolume"
    ob.scale = (130.0, 130.0, 30.0)
    ob.data.materials.append(m)
    ob.visible_shadow = False
    return ob


def add_camera(name, lens, sensor=36.0):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.sensor_width = sensor
    cd.clip_start = 0.03
    cd.clip_end = 1200.0
    ob = bpy.data.objects.new(name, cd)
    bpy.context.collection.objects.link(ob)
    return ob


def aim(cam, target):
    d = (Vector(target) - cam.location)
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def sample_points(objs, step=11):
    pts = []
    for ob in objs:
        mw = ob.matrix_world
        vs = ob.data.vertices
        n = len(vs)
        for i in range(0, n, step):
            v = vs[i].co
            pts.append((mw[0][0] * v.x + mw[0][1] * v.y + mw[0][2] * v.z + mw[0][3],
                        mw[1][0] * v.x + mw[1][1] * v.y + mw[1][2] * v.z + mw[1][3],
                        mw[2][0] * v.x + mw[2][1] * v.y + mw[2][2] * v.z + mw[2][3]))
    return pts


def fit_silhouette(scene, cam, pts, az, cam_z, aim_z, margin=0.035, iters=4):
    """Frame the REAL projected silhouette (not the bounding box):
    project the sampled points, then solve distance + aim so the silhouette
    is centred and fills the frame."""
    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y
    lens = cam.data.lens
    sw = cam.data.sensor_width
    if res_y >= res_x:
        f_v = lens / sw
        f_h = lens / (sw * res_x / res_y)
    else:
        f_h = lens / sw
        f_v = lens / (sw * res_y / res_x)
    tgt = 0.5 * (1.0 - 2.0 * margin)

    d = 22.0
    target = Vector((0.0, 0.0, aim_z))
    for _ in range(iters):
        cam.location = Vector((math.cos(az) * d, math.sin(az) * d, cam_z))
        aim(cam, target)
        bpy.context.view_layer.update()
        inv = cam.matrix_world.inverted()
        m = cam.matrix_world.to_3x3()
        right = Vector((m[0][0], m[1][0], m[2][0]))
        upv = Vector((m[0][1], m[1][1], m[2][1]))
        umin = vmin = float("inf")
        umax = vmax = float("-inf")
        zsum = 0.0
        zc = 0
        for (px, py, pz) in pts:
            q = inv @ Vector((px, py, pz))
            z = -q.z
            if z < 0.05:
                continue
            u = q.x * f_h / z
            v = q.y * f_v / z
            if u < umin:
                umin = u
            if u > umax:
                umax = u
            if v < vmin:
                vmin = v
            if v > vmax:
                vmax = v
            zsum += z
            zc += 1
        if zc == 0:
            return d
        zavg = zsum / max(1, zc)
        # distance so the silhouette fits
        s = max((umax - umin) / (2 * tgt), (vmax - vmin) / (2 * tgt))
        d = d * s
        # recentre: if the silhouette centre sits above/beside the frame centre,
        # the aim point must move the SAME way (up / right).
        cu = (umin + umax) * 0.5
        cv = (vmin + vmax) * 0.5
        lim = 0.25 * d
        su = max(-lim, min(lim, cu * zavg / f_h))
        sv = max(-lim, min(lim, cv * zavg / f_v))
        target = target + right * su + upv * sv

    cam.location = Vector((math.cos(az) * d, math.sin(az) * d, cam_z))
    aim(cam, target)
    bpy.context.view_layer.update()
    return d


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def apply_leaf_attributes(ob, st):
    me = ob.data
    ca = me.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
    flat = array("f")
    for col, cnt in zip(st.leaf_cols, st.leaf_counts):
        flat.extend((col[0], col[1], col[2], 1.0) * cnt)
    if len(flat) == len(ca.data) * 4:
        ca.data.foreach_set("color", flat)
    ra = me.attributes.new(name="leafrand", type="FLOAT", domain="POINT")
    vals = array("f")
    for r, cnt in zip(st.leaf_rands, st.leaf_counts):
        vals.extend([r] * cnt)
    if len(vals) == len(ra.data):
        ra.data.foreach_set("value", vals)


def main():
    global TEMPLATES

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = "HIP"
    except Exception:
        pass
    prefs.get_devices()
    for d in prefs.devices:
        d.use = (d.type == "HIP")
    scene.cycles.samples = SAMPLES
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.010
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.cycles.transmission_bounces = 6
    scene.cycles.volume_bounces = 1
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    scene.cycles.blur_glossy = 1.4
    scene.cycles.volume_max_steps = 64
    scene.render.resolution_x = RENDER_W
    scene.render.resolution_y = RENDER_H
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    for look in ("AgX - Punchy", "Punchy", "AgX - Medium High Contrast", "None"):
        try:
            scene.view_settings.look = look
            break
        except Exception:
            continue
    try:
        scene.view_settings.view_transform = "AgX"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0

    # ---- leaf templates -------------------------------------------------
    rng = random.Random(SEED)
    TEMPLATES = []
    for k in range(6):
        L = rng.uniform(0.175, 0.265)
        W = L * rng.uniform(0.40, 0.56)
        TEMPLATES.append(make_leaf_template(SEED + 101 * k, L, W))
    log("leaf templates ready")

    bark_buf = Buf()
    leaf_buf = Buf()
    st = TreeStats(SEED + 7)
    build_tree(bark_buf, leaf_buf, st)
    log("branches=%d  leaves=%d" % (st.n_branches, st.n_leaves))

    bark_mat = make_bark_material()
    leaf_mat = make_foliage_material(True)
    litter_mat = make_foliage_material(False)
    ground_mat = make_ground_material()

    tree_ob = bark_buf.to_object("Tree_Bark", bark_mat, smooth=True)
    leaves_ob = leaf_buf.to_object("Tree_Leaves", leaf_mat, smooth=True)
    apply_leaf_attributes(leaves_ob, st)
    log("leaf mesh: %d verts / %d faces" % (len(leaves_ob.data.vertices), len(leaves_ob.data.polygons)))

    gbuf = Buf()
    build_ground(gbuf)
    ground_ob = gbuf.to_object("Forest_Floor", ground_mat, smooth=True)

    lbuf = Buf()
    st2 = TreeStats(SEED + 99)
    build_litter(lbuf, st2, count=5200)
    litter_ob = lbuf.to_object("Litter", litter_mat, smooth=True)
    apply_leaf_attributes(litter_ob, st2)

    ggrass = Buf()
    st3 = TreeStats(SEED + 1234)
    build_grass(ggrass, st3, count=620000, radius=30.0, clump_thr=-0.24)
    build_grass(ggrass, st3, count=90000, radius=72.0, tall=True, clump_thr=-0.05)
    grass_ob = ggrass.to_object("Grass", make_grass_material(), smooth=True)
    apply_leaf_attributes(grass_ob, st3)
    log("grass: %d blades" % st3.n_leaves)

    # ---- world / lights -------------------------------------------------
    cam_az = math.radians(318.0)
    sun_az = cam_az + math.radians(164.0)
    sun_el = math.radians(9.0)
    setup_world(sun_az, sun_el)
    add_sun(sun_az, sun_el)
    add_fill(cam_az)
    add_bounce()
    if USE_MIST:
        add_mist()

    # ---- cameras --------------------------------------------------------
    pts = sample_points([tree_ob, leaves_ob], step=9)
    log("silhouette samples: %d" % len(pts))
    hero = add_camera("Cam_Hero", 70.0)
    d = fit_silhouette(scene, hero, pts, cam_az, cam_z=0.95, aim_z=4.2, margin=0.022)
    log("hero distance %.2f m" % d)
    scene.camera = hero

    trunk_cam = add_camera("Cam_Trunk", 78.0)
    ta = cam_az + math.radians(9.0)
    trunk_cam.location = Vector((math.cos(ta) * 5.20, math.sin(ta) * 5.20, 1.45))
    aim(trunk_cam, Vector((0.0, 0.0, 1.75)))
    trunk_cam.data.dof.use_dof = True
    trunk_cam.data.dof.focus_distance = 5.35
    trunk_cam.data.dof.aperture_fstop = 4.5

    can_cam = add_camera("Cam_Canopy", 62.0)
    ca2 = cam_az + math.radians(16.0)
    can_cam.location = Vector((math.cos(ca2) * 6.10, math.sin(ca2) * 6.10, 5.40))
    aim(can_cam, Vector((math.cos(ca2) * 2.90, math.sin(ca2) * 2.90, 5.05)))
    can_cam.data.dof.use_dof = True
    can_cam.data.dof.focus_distance = 3.25
    can_cam.data.dof.aperture_fstop = 3.5

    if CAM_NAME == "trunk":
        scene.camera = trunk_cam
    elif CAM_NAME == "canopy":
        scene.camera = can_cam

    if SAVE_BLEND:
        bp = os.path.join(OUT_DIR, "hero_tree.blend")
        bpy.ops.wm.save_as_mainfile(filepath=bp)
        log("saved %s" % bp)

    if DO_RENDER:
        scene.render.filepath = os.path.join(OUT_DIR, "render_%s.png" % CAM_NAME)
        log("rendering %s  %dx%d  %d spp" % (CAM_NAME, RENDER_W, RENDER_H, SAMPLES))
        bpy.ops.render.render(write_still=True)
        log("rendered -> %s" % scene.render.filepath)

    log("done")


if __name__ == "__main__":
    main()

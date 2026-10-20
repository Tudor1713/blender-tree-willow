# -*- coding: utf-8 -*-
"""
Weeping willow (垂柳) for Blender 5.2 -- procedural, shares the verified
material / lighting / ground / framing toolkit with build_tree.py.

Species specific design
  * 5 leaning stems from a shared root flare (not a single bole)
  * limbs that RISE, then arch over and turn downward
  * long pendulous shoots hanging 1.4-3.2 m, carrying narrow lanceolate
    leaves the whole way down -- the signature "curtain"
  * narrow willow leaves (width ~0.16 x length) with a raised midrib,
    fine side veins, a cupped cross section and a slight twist
"""

import bpy
import math
import random
import os
import sys
import time
from mathutils import Vector, Matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import build_tree as BT          # shared toolkit (materials, lights, ground, framing)

T0 = time.time()


def log(msg):
    print("[willow] %7.1fs  %s" % (time.time() - T0, msg), flush=True)


SEED = int(BT.arg("--seed", 777001))
OUT_DIR = BT.arg("--out", os.path.join(SCRIPT_DIR, "tree_output", "willow"))
RW = int(BT.arg("--rw", 2000))
RH = int(BT.arg("--rh", 2500))
SPP = int(BT.arg("--spp", 256))
CAM_NAME = BT.arg("--cam", "hero")
USE_MIST = not BT.flag("--nomist")
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# Species constants
# --------------------------------------------------------------------------
N_STEMS = 5
STEM_H = (3.55, 5.00)          # stem height range
STEM_R = (0.125, 0.195)         # stem base radius
LIMB_LEN = (2.40, 3.70)
PEND_LEN = (1.40, 3.05)
PEND_R = (0.0090, 0.0020)
W_MAX_H = 7.70                  # limbs must not climb above this
ROOT_R = 0.36
PEND_SPACING = 0.026
RNG = random.Random(SEED + 991)


# --------------------------------------------------------------------------
# Willow leaf: narrow lanceolate, pointed tip, prominent midrib
# --------------------------------------------------------------------------
W_ROWS = [0.0, 0.12, 0.28, 0.45, 0.62, 0.78, 0.90, 1.0]
W_COLS = 3


def w_profile(t):
    """Lanceolate: widest at roughly 1/3, drawn out to a fine point."""
    if t <= 0.0 or t >= 1.0:
        return 0.0
    return max(0.0, math.sin(math.pi * (t ** 0.55)) ** 1.30)


def make_willow_leaf(seed, length, width):
    rng = random.Random(seed)
    vein_ts = [0.16, 0.30, 0.44, 0.58, 0.72]
    vein_amp = 0.030 * length * rng.uniform(0.85, 1.15)
    midrib_h = 0.070 * length
    cup = 0.150 * length
    droop = 0.170 * length
    twist = rng.uniform(-0.55, 0.55)
    asym = rng.uniform(-0.18, 0.18)

    def relief(x, y, t, u):
        z = -cup * (u * u)
        z += midrib_h * math.exp(-((x / (0.075 * width)) ** 2))
        z -= droop * (t ** 1.9)
        for vt in vein_ts:
            wv = width * 0.5 * w_profile(vt)
            for sgn in (-1.0, 1.0):
                ax, ay = 0.0, vt * length
                bx, by = sgn * wv * 0.94, (vt + 0.15) * length
                d = BT._seg_dist(x, y, ax, ay, bx, by)
                z += vein_amp * math.exp(-((d / (0.045 * length)) ** 2))
        z += 0.008 * length * BT.fbm(Vector((x * 60.0, y * 30.0, 7.7)), 2)
        return z

    rows = []
    for ri, t in enumerate(W_ROWS):
        if ri == 0 or ri == len(W_ROWS) - 1:
            rows.append([(0.0, t * length, relief(0.0, t * length, t, 0.0))])
        else:
            w = width * 0.5 * w_profile(t)
            cx = asym * width * 0.30 * math.sin(math.pi * t)
            row = []
            for j in range(W_COLS):
                u = -1.0 + 2.0 * j / (W_COLS - 1)
                x = cx + u * w
                y = t * length
                row.append((x, y, relief(x, y, t, u) + twist * 0.30 * length * u * t))
            rows.append(row)

    verts = []
    idx = []
    for row in rows:
        b = len(verts)
        idx.append(list(range(b, b + len(row))))
        for v in row:
            verts.append(v)

    faces = []
    for j in range(W_COLS - 1):
        faces.append((idx[0][0], idx[1][j], idx[1][j + 1]))
    for ri in range(1, len(rows) - 2):
        A, B = idx[ri], idx[ri + 1]
        for j in range(W_COLS - 1):
            faces.append((A[j], B[j], B[j + 1], A[j + 1]))
    tip = idx[-1][0]
    for j in range(W_COLS - 1):
        faces.append((idx[-2][j], tip, idx[-2][j + 1]))
    return verts, faces


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def w_limit(P, d):
    """Keep limbs from climbing: they must arch over, not spear upward."""
    if P.z > W_MAX_H and d.z > 0.0:
        d = d - Vector((0.0, 0.0, d.z * 1.6))
    return d.normalized()


def grow_stem(buf, start, direction, height, r0, r1, lean_dir, lean_amt):
    """A willow stem: rises while leaning out, with a soft S-curve."""
    rng = RNG
    npts = 26
    path = []
    radii = []
    p = start.copy()
    d = direction.normalized()
    seg = height / (npts - 1)
    side = Vector((-lean_dir.y, lean_dir.x, 0.0))
    for i in range(npts):
        t = i / (npts - 1)
        path.append(p.copy())
        radii.append(r0 + (r1 - r0) * (t ** 0.9))
        d = d + lean_dir * (lean_amt * seg * 0.09)
        d = d + side * (math.sin(t * 5.2) * 0.035)
        d = d + Vector((0.0, 0.0, -0.04))
        d = d.normalized()
        p = p + d * seg
    BT.add_tube(buf, path, radii, 16, bark=0.13, seedoff=rng.uniform(0, 40))
    return path, radii


def grow_limb(buf, start, direction, length, r0, r1):
    """Arching limb: springs up-and-out, then curves over and points down."""
    rng = RNG
    npts = max(8, int(length * 8))
    path = []
    radii = []
    p = start.copy()
    d = direction.normalized()
    seg = length / (npts - 1)
    for i in range(npts):
        t = i / (npts - 1)
        path.append(p.copy())
        radii.append(r0 + (r1 - r0) * (t ** 0.85))
        # progressive arch: pitch the direction down as the limb extends
        d = d + Vector((0.0, 0.0, -(0.30 + 1.55 * t) * seg * 0.85))
        d = d + Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-0.4, 0.4))) * 0.030
        d = w_limit(p, d.normalized())
        p = p + d * seg
    BT.add_tube(buf, path, radii, 9, bark=0.13, seedoff=rng.uniform(0, 40))
    return path, radii


def grow_pendulous(buf, lbuf, st, start, direction, length, r0, r1):
    """The signature willow shoot: hangs almost straight down, curving gently
    outward at the top, carrying narrow leaves the whole way."""
    rng = st.rng
    # the path must be sampled finely enough to carry a leaf every
    # PEND_SPACING metres, otherwise the shoot can only hold a few leaves
    npts = max(18, int(length * 30))
    path = []
    radii = []
    p = start.copy()
    d = direction.normalized()
    seg = length / (npts - 1)
    outward = Vector((d.x, d.y, 0.0))
    if outward.length < 1e-5:
        outward = Vector((1.0, 0.0, 0.0))
    outward.normalize()
    for i in range(npts):
        t = i / (npts - 1)
        path.append(p.copy())
        radii.append(r0 + (r1 - r0) * (t ** 0.65))
        # swing out at the top, then fall straight
        d = d + outward * (0.16 * (1.0 - t) * seg * 0.9)
        d = d + Vector((0.0, 0.0, -0.10 * seg))
        d = d + Vector((rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35), 0.0)) * 0.012
        d = d.normalized()
        p = p + d * seg
    BT.add_tube(buf, path, radii, 5, bark=0.0, seedoff=rng.uniform(0, 40))

    # a few leaves sit directly on the limb so the bare arcs are not exposed
    for li in range(4, len(path) - 1, 3):
        lp = path[li]
        out = Vector((lp.x, lp.y, 0.0))
        if out.length < 1e-5:
            out = Vector((1.0, 0.0, 0.0))
        out.normalize()
        fwd = (out * 0.55 + Vector((0.0, 0.0, -0.78))).normalized()
        hsv = (rng.uniform(0.190, 0.280), rng.uniform(0.48, 0.90), rng.uniform(0.20, 0.60))
        BT.emit_leaf(lbuf, st, lp, fwd, Vector((0.0, 0.0, 1.0)), rng.uniform(0.80, 1.15),
                     hsv, rng)
        st.n_leaves += 1

    # --- leaves along the shoot -----------------------------------------
    arc = [0.0]
    for i in range(1, len(path)):
        arc.append(arc[-1] + (path[i] - path[i - 1]).length)
    total = arc[-1]
    if total < 1e-4:
        return
    spacing = PEND_SPACING
    i = 0
    side_sign = 1.0
    while i < npts - 1:
        p = path[i]
        t = arc[i] / total
        if t > 0.03:
            tv = path[i + 1] - path[i]
            if tv.length > 1e-8:
                tv.normalize()
                side = tv.cross(Vector((0.0, 0.0, 1.0)))
                if side.length < 1e-6:
                    side = Vector((1.0, 0.0, 0.0))
                side.normalize()
                out = Vector((p.x, p.y, 0.0))
                if out.length < 1e-5:
                    out = Vector((1.0, 0.0, 0.0))
                out.normalize()
                sgn = side_sign
                side_sign = -side_sign
                # willow leaves hang: mostly downward with an outward tilt
                fwd = (Vector((0.0, 0.0, -1.0)) * 0.52 + side * (0.46 * sgn)
                       + out * 0.30 + tv * 0.22).normalized()
                anchor = p + side * (sgn * 0.004) + out * 0.004
                sc = rng.uniform(0.80, 1.18)
                hsv = (rng.uniform(0.190, 0.280), rng.uniform(0.48, 0.90), rng.uniform(0.20, 0.60))
                BT.emit_leaf(lbuf, st, anchor, fwd, Vector((0.0, 0.0, 1.0)), sc, hsv, rng)
                st.n_leaves += 1
        nxt = i
        while nxt < npts - 1 and arc[nxt] < arc[i] + spacing:
            nxt += 1
        i = max(nxt, i + 1)
    return path, radii


def build_willow(buf, lbuf, st):
    rng = st.rng
    golden = math.radians(137.507)

    # --- shared root flare -----------------------------------------------
    root_path = []
    root_radii = []
    for i in range(9):
        t = i / 8.0
        root_path.append(Vector((0.0, 0.0, t * 0.55)))
        root_radii.append(ROOT_R * (1.0 - 0.22 * t))
    BT.add_tube(buf, root_path, root_radii, 20, bark=0.15, seedoff=7.0, cap=True)

    # buttress roots around the flare
    for k in range(6):
        a = golden * k
        rdir = Vector((math.cos(a), math.sin(a), 0.0))
        p = Vector((0.0, 0.0, 0.30))
        npts = 7
        path = []
        radii = []
        pp = p.copy()
        dd = (rdir * 0.85 + Vector((0.0, 0.0, -0.35))).normalized()
        for i in range(npts):
            tt = i / (npts - 1)
            path.append(pp.copy())
            radii.append(0.13 * (1.0 - 0.55 * tt))
            dd = (dd + Vector((0.0, 0.0, -0.10))).normalized()
            pp = pp + dd * (0.30 / (npts - 1))
        BT.add_tube(buf, path, radii, 8, bark=0.17, seedoff=60.0 + k * 5.0, cap=True)

    # --- stems -------------------------------------------------------------
    stems = []
    for k in range(N_STEMS):
        a = golden * k + rng.uniform(-0.16, 0.16)
        rdir = Vector((math.cos(a), math.sin(a), 0.0))
        lean_amt = rng.uniform(0.30, 0.72)
        start = rdir * rng.uniform(0.05, 0.16) + Vector((0.0, 0.0, 0.18))
        d = (rdir * rng.uniform(0.10, 0.26) + Vector((0.0, 0.0, 1.0))).normalized()
        h = rng.uniform(*STEM_H)
        path, radii = grow_stem(buf, start, d, h, rng.uniform(*STEM_R), rng.uniform(*STEM_R) * 0.55,
                                rdir, lean_amt)
        stems.append((path, radii, rdir))

    # --- limbs + pendulous shoots -----------------------------------------
    n_limb = 7
    for si, (spath, sradii, sdir) in enumerate(stems):
        for li in range(n_limb):
            t = 0.42 + 0.58 * (li + 0.5) / n_limb + rng.uniform(-0.05, 0.05)
            idx = max(1, min(len(spath) - 2, int(t * (len(spath) - 1))))
            P = spath[idx]
            az = golden * (li + 1 + si * 0.7) + rng.uniform(-0.30, 0.30)
            radial = Vector((math.cos(az), math.sin(az), 0.0))
            # limbs spring out and up, then arch over into a dome; kept low
            # enough that the leaf curtain covers the crown instead of
            # exposing bare arcs above it
            d = (radial * 0.62 + Vector((0.0, 0.0, 0.74))).normalized()
            L = rng.uniform(*LIMB_LEN) * (1.0 - 0.16 * (li / n_limb))
            r0 = sradii[idx] * 0.52
            lpath, lradii = grow_limb(buf, P, d, L, r0, r0 * 0.28)

            # secondary branches from the outer half of the limb
            n_sec = 3
            for sj in range(n_sec):
                ts = rng.uniform(0.32, 0.95)
                j = max(1, min(len(lpath) - 2, int(ts * (len(lpath) - 1))))
                Q = lpath[j]
                saz = golden * (sj + 2 + li * 0.5) + rng.uniform(-0.4, 0.4)
                srad = Vector((math.cos(saz), math.sin(saz), 0.0))
                sd = (srad * 0.62 + Vector((0.0, 0.0, 0.52))).normalized()
                SL = rng.uniform(1.05, 2.00)
                spath2, sradii2 = grow_limb(buf, Q, sd, SL, lradii[j] * 0.58, lradii[j] * 0.28)

                # dense curtain of pendulous shoots off each secondary branch
                n_pend = 10
                for pk in range(n_pend):
                    pt = rng.uniform(0.10, 0.99)
                    pi = max(1, min(len(spath2) - 2, int(pt * (len(spath2) - 1))))
                    R = spath2[pi]
                    paz = rng.uniform(0.0, 2.0 * math.pi)
                    prad = Vector((math.cos(paz), math.sin(paz), 0.0))
                    pd = (prad * 0.30 + Vector((0.0, 0.0, -1.0))).normalized()
                    PL = rng.uniform(*PEND_LEN) * (1.0 - 0.35 * pt)
                    grow_pendulous(buf, lbuf, st, R, pd, PL,
                                    sradii2[pi] * 0.40, max(0.0014, sradii2[pi] * 0.14))

            # pendulous shoots all along the outer limb -> the hanging skirt
            n_pend = 12
            for pk in range(n_pend):
                pt = 0.15 + 0.85 * (pk + rng.uniform(0.0, 0.8)) / n_pend
                pi = max(1, min(len(lpath) - 2, int(pt * (len(lpath) - 1))))
                R = lpath[pi]
                paz = rng.uniform(0.0, 2.0 * math.pi)
                prad = Vector((math.cos(paz), math.sin(paz), 0.0))
                pd = (prad * 0.28 + Vector((0.0, 0.0, -1.0))).normalized()
                PL = rng.uniform(*PEND_LEN) * (0.60 + 0.70 * pt)
                grow_pendulous(buf, lbuf, st, R, pd, PL,
                                lradii[pi] * 0.38, max(0.0014, lradii[pi] * 0.13))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
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
    scene.cycles.samples = SPP
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
    scene.render.resolution_x = RW
    scene.render.resolution_y = RH
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    for look in ("AgX - Punchy", "Punchy", "None"):
        try:
            scene.view_settings.look = look
            break
        except Exception:
            continue
    try:
        scene.view_settings.view_transform = "AgX"
    except Exception:
        pass

    # --- willow leaf templates -------------------------------------------
    rng = random.Random(SEED)
    BT.TEMPLATES = []
    for k in range(6):
        L = rng.uniform(0.115, 0.175)
        W = L * rng.uniform(0.165, 0.220)
        BT.TEMPLATES.append(make_willow_leaf(SEED + 211 * k, L, W))
    log("willow leaf templates ready")

    bark_buf = BT.Buf()
    leaf_buf = BT.Buf()
    st = BT.TreeStats(SEED + 3)
    build_willow(bark_buf, leaf_buf, st)
    log("branches=%d leaves=%d" % (st.n_branches, st.n_leaves))

    bark_mat = BT.make_bark_material()
    leaf_mat = BT.make_foliage_material(True)
    ground_mat = BT.make_ground_material()

    tree_ob = bark_buf.to_object("Willow_Bark", bark_mat, smooth=True)
    leaves_ob = leaf_buf.to_object("Willow_Leaves", leaf_mat, smooth=True)
    BT.apply_leaf_attributes(leaves_ob, st)
    log("leaf mesh: %d verts / %d faces" % (len(leaves_ob.data.vertices), len(leaves_ob.data.polygons)))

    gbuf = BT.Buf()
    BT.build_ground(gbuf)
    ground_ob = gbuf.to_object("Forest_Floor", ground_mat, smooth=True)

    lbuf = BT.Buf()
    st2 = BT.TreeStats(SEED + 991)
    BT.build_litter(lbuf, st2, count=4200)
    litter_ob = lbuf.to_object("Litter", BT.make_foliage_material(False), smooth=True)
    BT.apply_leaf_attributes(litter_ob, st2)

    ggrass = BT.Buf()
    st3 = BT.TreeStats(SEED + 4242)
    BT.build_grass(ggrass, st3, count=560000, radius=30.0, clump_thr=-0.24)
    BT.build_grass(ggrass, st3, count=80000, radius=72.0, tall=True, clump_thr=-0.05)
    grass_ob = ggrass.to_object("Grass", BT.make_grass_material(), smooth=True)
    BT.apply_leaf_attributes(grass_ob, st3)
    log("grass: %d blades" % st3.n_leaves)

    # --- world / lights ---------------------------------------------------
    cam_az = math.radians(318.0)
    sun_az = cam_az + math.radians(158.0)
    sun_el = math.radians(11.5)
    BT.setup_world(sun_az, sun_el)
    BT.add_sun(sun_az, sun_el)
    BT.add_fill(cam_az)
    BT.add_bounce()
    if USE_MIST:
        BT.add_mist()

    # --- cameras ----------------------------------------------------------
    pts = BT.sample_points([tree_ob, leaves_ob], step=9)
    log("silhouette samples: %d" % len(pts))
    hero = BT.add_camera("Cam_Hero", 66.0)
    d = BT.fit_silhouette(scene, hero, pts, cam_az, cam_z=1.10, aim_z=5.0, margin=0.025)
    log("hero distance %.2f m" % d)
    scene.camera = hero

    # curtain detail: stand OUTSIDE the crown and shoot the hanging strands
    # against the bright sky, otherwise the camera ends up buried in foliage
    can_cam = BT.add_camera("Cam_Canopy", 88.0)
    ca2 = cam_az + math.radians(24.0)
    can_cam.location = Vector((math.cos(ca2) * 10.20, math.sin(ca2) * 10.20, 5.10))
    aim_target = Vector((math.cos(ca2) * 4.30, math.sin(ca2) * 4.30, 3.10))
    BT.aim(can_cam, aim_target)
    can_cam.data.dof.use_dof = True
    can_cam.data.dof.focus_distance = 6.10
    can_cam.data.dof.aperture_fstop = 5.6

    trunk_cam = BT.add_camera("Cam_Trunk", 72.0)
    ta = cam_az + math.radians(10.0)
    trunk_cam.location = Vector((math.cos(ta) * 4.60, math.sin(ta) * 4.60, 1.35))
    BT.aim(trunk_cam, Vector((0.0, 0.0, 1.60)))
    trunk_cam.data.dof.use_dof = True
    trunk_cam.data.dof.focus_distance = 4.8
    trunk_cam.data.dof.aperture_fstop = 4.0

    if CAM_NAME == "canopy":
        scene.camera = can_cam
    elif CAM_NAME == "trunk":
        scene.camera = trunk_cam

    bp = os.path.join(OUT_DIR, "willow.blend")
    bpy.ops.wm.save_as_mainfile(filepath=bp)
    log("saved %s" % bp)

    if not BT.flag("--norender"):
        scene.render.filepath = os.path.join(OUT_DIR, "willow_%s.png" % CAM_NAME)
        log("rendering %s %dx%d %d spp" % (CAM_NAME, RW, RH, SPP))
        bpy.ops.render.render(write_still=True)
        log("rendered -> %s" % scene.render.filepath)
    log("done")


if __name__ == "__main__":
    main()

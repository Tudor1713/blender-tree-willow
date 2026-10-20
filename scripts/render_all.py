# -*- coding: utf-8 -*-
"""
Batch renderer -- opens the saved hero_tree.blend and renders every camera
setup (hero / trunk / canopy) at the requested resolution and sample count.

    blender -b hero_tree.blend -P render_all.py -- --rw 2000 --rh 2500 --spp 256
"""
import bpy
import os
import sys
import time

RAW = [a for a in sys.argv[1:]]


def arg(name, default=None):
    for i, a in enumerate(RAW):
        if a == name and i + 1 < len(RAW):
            return RAW[i + 1]
    return default


def flag(name):
    return name in RAW


OUT = arg("--out", os.path.join(os.path.dirname(bpy.data.filepath), "renders"))
RW = int(arg("--rw", 2000))
RH = int(arg("--rh", 2500))
SPP = int(arg("--spp", 256))
ONLY = arg("--cam", None)

os.makedirs(OUT, exist_ok=True)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "GPU"
prefs = bpy.context.preferences.addons["cycles"].preferences
for d in prefs.devices:
    d.use = (d.type == "HIP")
scene.cycles.samples = SPP
scene.cycles.use_denoising = True
scene.render.resolution_x = RW
scene.render.resolution_y = RH
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"

# per-camera framing (the hero shot is portrait, the detail shots are landscape)
PLAN = [
    ("hero", "Cam_Hero", RW, RH, 0.030, None),
    ("trunk", "Cam_Trunk", int(RW * 1.30), int(RW * 0.80), 0.0, None),
    ("canopy", "Cam_Canopy", int(RW * 1.30), int(RW * 0.80), 0.0, None),
]

for name, cam_name, w, h, margin, _ in PLAN:
    if ONLY and name != ONLY:
        continue
    cam = bpy.data.objects.get(cam_name)
    if cam is None:
        print("[batch] missing camera", cam_name)
        continue
    scene.camera = cam
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.filepath = os.path.join(OUT, "tree_%s.png" % name)
    t0 = time.time()
    print("[batch] rendering %s  %dx%d  %d spp -> %s" % (name, w, h, SPP, scene.render.filepath),
          flush=True)
    bpy.ops.render.render(write_still=True)
    print("[batch] done %s in %.1fs" % (name, time.time() - t0), flush=True)

print("[batch] all done")

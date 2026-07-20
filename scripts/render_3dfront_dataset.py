#!/usr/bin/env python3
"""Production batch renderer: 3D-FRONT rooms -> CARI cross-illuminant IID dataset.

Renders, per room and per sampled interior view, K same-camera lighting
variants plus a ground-truth albedo:

  <out>/<house>/<room>/view_00/
      rgb_L0.png  rgb_L0.exr      (lit render, sRGB preview + linear)
      rgb_L1.png  rgb_L1.exr
      albedo.png  albedo.exr      (material base color via emission trick)
      meta.json                   (camera, light colors/positions, seeds)

Findings baked in from the single-room pilots (tests/viz/3dfront_iid_pair*):
  - MIDI meta.json cameras are exterior dollhouse poses -> unusable for
    photo-like training views. Cameras are sampled INSIDE the room instead,
    validated by ray casting (coverage / not-inside-furniture / not staring
    at one wall), BlenderProc-front3d style.
  - Room GLBs can have open faces, so a neutral shell (randomly tinted per
    room) backstops missing walls.
  - The two fixed warm/cool configs of the pilot would give CARI a single
    illuminant delta repeated 1000x. Lighting here is randomized per variant
    (blackbody 2500-9000K or saturated HSV) with a minimum rg-chromaticity
    gap enforced between the K key lights of a view, so L_inv pairs always
    see a real illuminant-color change.
  - Each output is rendered ONCE; PNG and EXR are both written from the same
    Render Result via save_render (the pilot re-rendered for each format).

Prep (one-time; per-room tar streaming over the 13GB archive is not viable):
  mkdir -p ~/datasets/3D-Front-HF/scenes
  tar xzf ~/datasets/3D-Front-HF/3D-FRONT-TEST-SCENE.tar.gz -C ~/datasets/3D-Front-HF/scenes

Run (shard across GPUs with --start/--end + CUDA_VISIBLE_DEVICES):
  python3 scripts/render_3dfront_dataset.py \
      --glb-root ~/datasets/3D-Front-HF/scenes/3D-FRONT-TEST-SCENE \
      --out ~/datasets/front3d_iid --device GPU --views 3 --lightings 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path


BLENDER_SCRIPT = r"""
import argparse
import colorsys
import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv
    argv = argv[argv.index('--') + 1:] if '--' in argv else []
    p = argparse.ArgumentParser()
    p.add_argument('--glb', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--room-id', required=True)
    p.add_argument('--views', type=int, default=3)
    p.add_argument('--lightings', type=int, default=2)
    p.add_argument('--resolution', type=int, default=512)
    p.add_argument('--samples', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', choices=['CPU', 'GPU'], default='CPU')
    return p.parse_args(argv)


# ── scene setup ─────────────────────────────────────────────────────────────

def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def import_glb(glb):
    bpy.ops.import_scene.gltf(filepath=str(glb))
    for obj in list(bpy.context.scene.objects):
        if obj.type in {'LIGHT', 'CAMERA'}:
            bpy.data.objects.remove(obj, do_unlink=True)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not meshes:
        raise RuntimeError(f'No mesh objects imported from {glb}')
    return meshes


def is_content_object_name(name):
    # True for furniture-like meshes used to accept photo-like views.
    # 3D-FRONT room GLBs often store architecture as None.obj/world-like nodes.
    # Counting those as content lets wall-close views pass the sampler.
    n = (name or '').lower()
    if n in {'', 'none', 'none.obj', 'world'}:
        return False
    if n.startswith('synthetic_'):
        return False
    architecture_tokens = (
        'wall', 'floor', 'ceil', 'ceiling', 'baseboard', 'skirting',
        'door', 'window', 'curtain', 'lighting',
    )
    return not any(tok in n for tok in architecture_tokens)


def scene_bounds(meshes):
    pts = []
    for obj in meshes:
        for corner in obj.bound_box:
            pts.append(obj.matrix_world @ Vector(corner))
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    center = (lo + hi) * 0.5
    size = hi - lo
    radius = max(float(size.x), float(size.y), float(size.z)) * 0.5
    return lo, hi, center, size, max(radius, 1.0)


def make_diffuse_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf:
        bsdf.inputs['Base Color'].default_value = color
        bsdf.inputs['Roughness'].default_value = 0.72
    return mat


def add_box(name, location, scale, material):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    return obj


def add_room_shell(rng, lo, hi, center, size, radius):
    # Neutral backstop box around the room for GLBs with missing walls.
    # Albedos are jittered per room so the shell does not inject a constant
    # wall/floor color prior into the dataset.
    def tint(base, jitter=0.06):
        return tuple(min(1.0, max(0.05, c + rng.uniform(-jitter, jitter))) for c in base) + (1.0,)

    width = max(float(size.x) * 1.85, 3.2)
    depth = max(float(size.y) * 1.55, 3.2)
    height = max(float(size.z) * 1.75, 2.4)
    floor_z = float(lo.z) - 0.035
    wall_z = floor_z + height * 0.5
    thickness = max(0.035 * radius, 0.04)

    mats = {
        'floor': make_diffuse_material('synthetic_floor_albedo', tint((0.43, 0.36, 0.28))),
        'back': make_diffuse_material('synthetic_back_wall_albedo', tint((0.58, 0.58, 0.55))),
        'side': make_diffuse_material('synthetic_side_wall_albedo', tint((0.52, 0.54, 0.56))),
        'ceiling': make_diffuse_material('synthetic_ceiling_albedo', tint((0.62, 0.62, 0.60))),
    }
    shell = [
        add_box('synthetic_floor', center + Vector((0, 0, floor_z - thickness * 0.5 - center.z)), (width, depth, thickness), mats['floor']),
        add_box('synthetic_back_wall', center + Vector((0, depth * 0.5, wall_z - center.z)), (width, thickness, height), mats['back']),
        add_box('synthetic_front_wall', center + Vector((0, -depth * 0.5, wall_z - center.z)), (width, thickness, height), mats['back']),
        add_box('synthetic_left_wall', center + Vector((-width * 0.5, 0, wall_z - center.z)), (thickness, depth, height), mats['side']),
        add_box('synthetic_right_wall', center + Vector((width * 0.5, 0, wall_z - center.z)), (thickness, depth, height), mats['side']),
        add_box('synthetic_ceiling', center + Vector((0, 0, floor_z + height + thickness * 0.5 - center.z)), (width, depth, thickness), mats['ceiling']),
    ]
    return shell


def setup_render(resolution, samples, device):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    if scene.world is None:
        scene.world = bpy.data.worlds.new('World')
    scene.world.use_nodes = False
    scene.world.color = (0.0, 0.0, 0.0)

    if device == 'GPU':
        scene.cycles.device = 'GPU'
        prefs = bpy.context.preferences.addons['cycles'].preferences
        for backend in ('OPTIX', 'CUDA'):
            try:
                prefs.compute_device_type = backend
                break
            except Exception:
                continue
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type != 'CPU'


# ── interior camera sampling ────────────────────────────────────────────────

def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def probe_view(cam, content_names, lo, hi, grid=12):
    # Ray-cast a grid through the camera frustum and collect framing metrics.
    # Gate decisions live in view_gates(); scoring in view_score().
    depsgraph = bpy.context.evaluated_depsgraph_get()
    scene = bpy.context.scene
    tan_x = cam.data.sensor_width / (2.0 * cam.data.lens)
    tan_y = tan_x  # square renders
    origin = cam.location.copy()
    rot = cam.matrix_world.to_3x3()

    n = hit = content = center_n = center_content = near = ceil_hits = 0
    dists = []
    objects = set()
    content_objects = set()
    ceil_thresh = float(lo.z) + 0.72 * float(hi.z - lo.z)
    for iy in range(grid):
        for ix in range(grid):
            u = (ix + 0.5) / grid * 2.0 - 1.0
            v = (iy + 0.5) / grid * 2.0 - 1.0
            is_center_ray = abs(u) <= 0.50 and abs(v) <= 0.50
            if is_center_ray:
                center_n += 1
            d_cam = Vector((u * tan_x, v * tan_y, -1.0)).normalized()
            d = (rot @ d_cam).normalized()
            n += 1
            ok, loc, _nrm, _idx, obj, _mat = scene.ray_cast(depsgraph, origin, d)
            if not ok:
                continue
            hit += 1
            dist = (loc - origin).length
            dists.append(dist)
            objects.add(obj.name)
            if obj.name in content_names:
                content += 1
                content_objects.add(obj.name)
                if is_center_ray:
                    center_content += 1
            if dist < 0.45:
                near += 1
            if loc.z > ceil_thresh:
                ceil_hits += 1

    if not dists:
        return None
    dists.sort()
    return {
        'hit': hit / n, 'content': content / n,
        'center_content': center_content / max(1, center_n),
        'near': near / n, 'ceil': ceil_hits / n,
        'objects': len(objects), 'content_objects': len(content_objects),
        'min_dist': dists[0], 'med_dist': dists[len(dists) // 2],
        'spread': dists[int(0.9 * len(dists))] - dists[int(0.1 * len(dists))],
    }


def view_reject_reason(m, strict=True):
    # Strict limits first; the relaxed tier keeps small/cluttered rooms
    # usable instead of dropping them from the dataset.
    i = 0 if strict else 1
    if m['hit'] < (0.92, 0.85)[i]:
        return 'void'
    if m['min_dist'] < (0.18, 0.10)[i]:
        return 'embedded'
    if m['near'] > (0.22, 0.32)[i]:
        return 'near_surface'
    if m['content'] < (0.10, 0.06)[i]:
        return 'furniture_sparse'
    if m['center_content'] < (0.05, 0.03)[i]:
        return 'off_object'
    if m['content_objects'] < 1:
        return 'blank_wall'
    if m['ceil'] > (0.35, 0.50)[i]:
        return 'ceiling'
    if m['med_dist'] < (0.90, 0.70)[i]:
        return 'too_close'
    return None


def view_score(m):
    return (1.6 * m['content'] + 0.8 * m['center_content']
            + 0.08 * min(m['spread'], 4.0)
            - 1.8 * m['near'] - 0.8 * m['ceil'])


def sample_interior_cameras(rng, lo, hi, size, content_names, num_views, tries=120):
    cam_data = bpy.data.cameras.new('Camera')
    cam_data.lens = 24.0
    cam_data.sensor_width = 36.0
    cam_data.clip_end = 200.0
    cam = bpy.data.objects.new('Camera', cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    bpy.context.view_layer.update()

    floor_z = float(lo.z)
    ceil_z = float(hi.z)
    strict_ok, relaxed_ok = [], []
    reject_counts = {}
    for _ in range(tries):
        margin = 0.22
        x = rng.uniform(lo.x + margin * size.x, hi.x - margin * size.x)
        y = rng.uniform(lo.y + margin * size.y, hi.y - margin * size.y)
        z = min(floor_z + rng.uniform(1.25, 1.75), ceil_z - 0.35)
        cam.location = Vector((x, y, z))
        tx = rng.uniform(lo.x + 0.3 * size.x, hi.x - 0.3 * size.x)
        ty = rng.uniform(lo.y + 0.3 * size.y, hi.y - 0.3 * size.y)
        # Level-or-down gaze: real photos rarely pitch up into the ceiling.
        tz = min(floor_z + rng.uniform(0.6, 1.2), z - 0.05)
        if (Vector((tx, ty, tz)) - cam.location).length < 0.8:
            continue
        look_at(cam, (tx, ty, tz))
        bpy.context.view_layer.update()
        m = probe_view(cam, content_names, lo, hi)
        if m is None:
            reject_counts['no_hits'] = reject_counts.get('no_hits', 0) + 1
            continue
        cand = (view_score(m), cam.location.copy(), cam.rotation_euler.copy(), dict(m))
        strict_reason = view_reject_reason(m, strict=True)
        if strict_reason is None:
            strict_ok.append(cand)
        else:
            reject_counts[strict_reason] = reject_counts.get(strict_reason, 0) + 1
            if view_reject_reason(m, strict=False) is None:
                relaxed_ok.append(cand)

    candidates = strict_ok if strict_ok else relaxed_ok
    if not candidates:
        print('camera rejection histogram:', json.dumps(reject_counts), file=sys.stderr)
        return cam, []
    if not strict_ok:
        print('camera sampling fell back to RELAXED gates;',
              'rejections:', json.dumps(reject_counts), file=sys.stderr)
    candidates.sort(key=lambda c: -c[0])
    picked = []
    for score, loc, rot, metrics in candidates:
        if all((loc - p[1]).length > 0.55 for p in picked):
            picked.append((score, loc, rot, metrics))
        if len(picked) == num_views:
            break
    return cam, picked


# ── randomized colored illuminants (the CARI axis) ──────────────────────────

def kelvin_to_rgb(k):
    # Tanner Helland blackbody approximation -> linear RGB.
    k = max(1500.0, min(15000.0, k)) / 100.0
    r = 255.0 if k <= 66 else 329.698727446 * ((k - 60) ** -0.1332047592)
    g = (99.4708025861 * math.log(k) - 161.1195681661) if k <= 66 else 288.1221695283 * ((k - 60) ** -0.0755148492)
    b = 255.0 if k >= 66 else (0.0 if k <= 19 else 138.5177312231 * math.log(k - 10) - 305.0447927307)
    srgb = [min(255.0, max(0.0, c)) / 255.0 for c in (r, g, b)]
    return tuple(c ** 2.2 for c in srgb)


def rg_chroma(color):
    s = sum(color) + 1e-8
    return (color[0] / s, color[1] / s)


def sample_key_color(rng):
    if rng.random() < 0.65:
        kelvin = math.exp(rng.uniform(math.log(2500.0), math.log(9000.0)))
        return kelvin_to_rgb(kelvin), {'type': 'blackbody', 'kelvin': round(kelvin)}
    # Saturation cap: past ~0.6 a lone key nearly monochromes the frame and
    # albedo stops being recoverable where a channel dies.
    h, s = rng.random(), rng.uniform(0.30, 0.60)
    srgb = colorsys.hsv_to_rgb(h, s, 1.0)
    return tuple(c ** 2.2 for c in srgb), {'type': 'hsv', 'hue': round(h, 3), 'sat': round(s, 3)}


def sample_lighting_set(rng, num_lightings, min_chroma_gap=0.055, tries=40):
    # K key colors with pairwise rg-chromaticity gaps so L_inv sees a real
    # illuminant change.
    keys = []
    for _ in range(num_lightings):
        for _attempt in range(tries):
            color, desc = sample_key_color(rng)
            c = rg_chroma(color)
            if all(math.dist(c, rg_chroma(k[0])) > min_chroma_gap for k in keys):
                keys.append((color, desc))
                break
        else:
            keys.append(sample_key_color(rng))  # accept a close pair over failing
    return keys


def clear_lights():
    for obj in list(bpy.context.scene.objects):
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)


def add_area(name, loc, target, color, power, size):
    data = bpy.data.lights.new(name, type='AREA')
    data.energy = power
    data.color = color[:3]
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = loc
    look_at(obj, target)
    return obj


def build_lighting(rng, lo, hi, center, size, radius, key_color):
    # One key + one fill, randomized placement in the upper room volume.
    clear_lights()
    floor_z = float(lo.z)
    height = float(size.z)

    def spot(zlo, zhi):
        return Vector((
            rng.uniform(lo.x + 0.18 * size.x, hi.x - 0.18 * size.x),
            rng.uniform(lo.y + 0.18 * size.y, hi.y - 0.18 * size.y),
            floor_z + rng.uniform(zlo, zhi) * height,
        ))

    target = Vector((center.x + rng.uniform(-0.25, 0.25) * size.x,
                     center.y + rng.uniform(-0.25, 0.25) * size.y,
                     floor_z + 0.35 * height))
    key_power = rng.uniform(1.2, 3.2) * radius * radius
    lights = [{
        'role': 'key', 'color': [round(c, 4) for c in key_color],
        'power': round(key_power, 1),
        'loc': spot(0.68, 0.92), 'size': rng.uniform(0.25, 0.55) * radius,
    }]
    # Always add a fill: a lone saturated key can kill a color channel
    # scene-wide, making albedo unrecoverable in the pair.
    if rng.random() < 0.7:
        fill_color = kelvin_to_rgb(rng.uniform(4500, 7000))
    else:
        fill_color, _ = sample_key_color(rng)
    lights.append({
        'role': 'fill', 'color': [round(c, 4) for c in fill_color],
        'power': round(key_power * rng.uniform(0.15, 0.35), 1),
        'loc': spot(0.45, 0.85), 'size': rng.uniform(0.6, 1.0) * radius,
    })

    cam = bpy.context.scene.camera
    if cam is not None:
        view_dir = (cam.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized()
        view_target = cam.location + view_dir * max(radius, 2.0)
        # Weak view-facing fill prevents fully black samples when the random
        # key/fill fall behind an interior wall. It is neutral and weaker than
        # the colored key, so the CARI illuminant change remains visible.
        lights.append({
            'role': 'view_fill',
            'color': [round(c, 4) for c in kelvin_to_rgb(6000)],
            'power': round(key_power * rng.uniform(0.14, 0.22), 1),
            'loc': cam.location - view_dir * 0.15 + Vector((0, 0, 0.08)),
            'size': max(0.8, 0.55 * radius),
            'target': view_target,
        })

    for i, l in enumerate(lights):
        add_area(f"{l['role']}_{i}", l['loc'], l.get('target', target), tuple(l['color']), l['power'], l['size'])
        l.pop('target', None)
        l['loc'] = [round(float(v), 3) for v in l['loc']]
        l['size'] = round(l['size'], 3)
    return lights


# ── rendering ───────────────────────────────────────────────────────────────

def render_once_save_both(out_dir, stem):
    # Render once; write linear EXR and sRGB PNG from the same result.
    scene = bpy.context.scene
    scene.render.filepath = str(out_dir / f'{stem}.exr')
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '32'
    bpy.ops.render.render(write_still=True)

    result = bpy.data.images.get('Render Result')
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_depth = '8'
    result.save_render(filepath=str(out_dir / f'{stem}.png'), scene=scene)


def material_to_emission():
    for mat in bpy.data.materials:
        mat.use_nodes = True
        tree = mat.node_tree
        output = next((n for n in tree.nodes if n.type == 'OUTPUT_MATERIAL'), None)
        if output is None:
            output = tree.nodes.new('ShaderNodeOutputMaterial')
        bsdf = next((n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
        emission = tree.nodes.new('ShaderNodeEmission')
        emission.inputs['Strength'].default_value = 1.0
        if bsdf and 'Base Color' in bsdf.inputs:
            base = bsdf.inputs['Base Color']
            if base.is_linked:
                tree.links.new(base.links[0].from_socket, emission.inputs['Color'])
            else:
                emission.inputs['Color'].default_value = base.default_value
        else:
            emission.inputs['Color'].default_value = mat.diffuse_color
        for link in list(tree.links):
            if link.to_node == output and link.to_socket == output.inputs['Surface']:
                tree.links.remove(link)
        tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])


def main():
    args = parse_args()
    out_root = Path(args.out)
    rng = random.Random(args.seed)

    clean_scene()
    meshes = import_glb(Path(args.glb))
    content_names = {m.name for m in meshes if is_content_object_name(m.name)}
    if not content_names:
        print(json.dumps({'room_id': args.room_id, 'status': 'no_content_mesh'}))
        return
    c_lo, c_hi, c_center, c_size, c_radius = scene_bounds(meshes)
    add_room_shell(rng, c_lo, c_hi, c_center, c_size, c_radius)
    setup_render(args.resolution, args.samples, args.device)
    bpy.context.view_layer.update()

    cam, views = sample_interior_cameras(rng, c_lo, c_hi, c_size, content_names, args.views)
    if not views:
        print(json.dumps({'room_id': args.room_id, 'status': 'no_valid_view'}))
        return

    view_records = []
    for vi, (score, loc, rot, metrics) in enumerate(views):
        cam.location, cam.rotation_euler = loc, rot
        view_dir = out_root / f'view_{vi:02d}'
        view_dir.mkdir(parents=True, exist_ok=True)
        keys = sample_lighting_set(rng, args.lightings)
        light_meta = []
        for li, (key_color, key_desc) in enumerate(keys):
            lights = build_lighting(rng, c_lo, c_hi, c_center, c_size, c_radius, key_color)
            render_once_save_both(view_dir, f'rgb_L{li}')
            light_meta.append({'key': key_desc, 'lights': lights})
        view_records.append({'dir': view_dir, 'score': score, 'metrics': metrics, 'lights': light_meta,
                             'cam_loc': [float(v) for v in loc],
                             'cam_rot': [float(v) for v in rot]})

    # Albedo pass last: emission conversion is destructive.
    clear_lights()
    bpy.context.scene.world.color = (0.0, 0.0, 0.0)
    material_to_emission()
    for vi, rec in enumerate(view_records):
        _score, loc, rot, _metrics = views[vi]
        cam.location, cam.rotation_euler = loc, rot
        render_once_save_both(rec['dir'], 'albedo')

    for rec in view_records:
        meta = {
            'room_id': args.room_id,
            'seed': args.seed,
            'camera': {'location': rec['cam_loc'], 'rotation_euler': rec['cam_rot'],
                       'lens': float(cam.data.lens), 'sensor_width': float(cam.data.sensor_width)},
            'view_score': round(rec['score'], 4),
            'view_metrics': {k: round(float(v), 4) for k, v in rec['metrics'].items()},
            'lightings': rec['lights'],
            'resolution': args.resolution,
            'samples': args.samples,
            'outputs': [f'rgb_L{i}.{ext}' for i in range(args.lightings) for ext in ('png', 'exr')] + ['albedo.png', 'albedo.exr'],
        }
        (rec['dir'] / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps({'room_id': args.room_id, 'status': 'ok', 'views': len(view_records)}))


if __name__ == '__main__':
    main()
"""


def find_rooms(glb_root: Path) -> list[Path]:
    return sorted(glb_root.rglob('*_full.glb'))


def room_id_of(glb: Path, glb_root: Path) -> str:
    rel = glb.relative_to(glb_root)
    stem = rel.stem[:-5] if rel.stem.endswith('_full') else rel.stem
    return str(rel.parent / stem)


def room_done(out_dir: Path, views: int, lightings: int) -> bool:
    # A room counts as done with ANY complete >=2-lighting view (not the current
    # --lightings): the loader needs one pair, and raising K on a rerun must not
    # re-render rooms already finished at a lower K (mixed-K corpora are fine).
    if not out_dir.is_dir():
        return False
    done_views = 0
    for vd in out_dir.glob('view_*'):
        if not (vd / 'meta.json').is_file():
            continue
        lit = [p for p in vd.glob('rgb_L*.exr')
               if p.stat().st_size > 0 and (vd / f'{p.stem}.png').is_file()]
        needed = ['albedo.png', 'albedo.exr']
        if len(lit) >= 2 and all((vd / n).is_file() and (vd / n).stat().st_size > 0 for n in needed):
            done_views += 1
    return done_views >= 1  # camera sampling may legitimately yield < --views


def validate_view(view_dir: Path, lightings: int) -> str | None:
    """Integrity + coarse quality check for generated PNG previews.

    Training uses EXR, but the PNG preview quickly catches bad camera/exposure
    regimes before a large render job fills the dataset with unusable pairs.
    """
    from PIL import Image
    import numpy as np

    hashes = {}
    arrays = {}
    # Derive K from disk, not from --lightings: mixed-K corpora (rooms rendered
    # at K=2 before a K=3 rerun) must check exactly the variants each view has.
    # Floor 2 = the minimum viable pair; fewer files fail the 'missing' check.
    lightings = max(2, len(list(view_dir.glob('rgb_L*.png'))))
    names = [f'rgb_L{i}.png' for i in range(lightings)] + ['albedo.png']
    for name in names:
        p = view_dir / name
        if not p.is_file():
            return f'missing {name}'
        hashes[name] = sha256(p.read_bytes()).hexdigest()
        arrays[name] = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0
    if len(set(hashes.values())) != len(hashes):
        return 'duplicate outputs (lighting variation or albedo pass failed)'

    meta_path = view_dir / 'meta.json'
    if meta_path.is_file():
        metrics = json.loads(meta_path.read_text()).get('view_metrics', {})
        if metrics:
            content = float(metrics.get('content', 0.0))
            center = float(metrics.get('center_content', 0.0))
            near = float(metrics.get('near', 0.0))
            if content < 0.06:
                return f'low furniture coverage ({content:.3f})'
            if center < 0.03:
                return f'off-object view ({center:.3f})'
            if near > 0.32:
                return f'too close to surface ({near:.3f})'

    def lum(x):
        return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]

    alb_l = lum(arrays['albedo.png'])
    valid = alb_l > 0.01
    coverage = float(valid.mean())
    if coverage < 0.75:
        return f'low valid coverage ({coverage:.3f})'

    # Dark-ALBEDO pixels (black wardrobes, dark fabric) are dark under ANY
    # illuminant; measuring `black` only where the material is bright makes it
    # detect lighting failure instead of rejecting valid dark furniture.
    bright = valid & (alb_l > 0.20)  # sRGB 0.20 ~ linear 0.03; below = truly dark material
    if float(bright.mean()) < 0.25:
        return f'dark-material dominated (bright={float(bright.mean()):.3f})'

    for i in range(lightings):
        name = f'rgb_L{i}.png'
        arr = arrays[name]
        rgb = arr[valid]
        l = lum(arr)
        clip = float((rgb.max(axis=-1) >= 0.995).mean())
        black = float((l[bright] <= 0.01).mean())
        if clip > 0.20:
            return f'{name} overexposed (clip={clip:.3f})'
        if black > 0.15:
            return f'{name} too dark/background-heavy (black={black:.3f})'
    return None


def quarantine_view(view_dir: Path, reason: str) -> None:
    """Drop ONE view from the dataset, keeping its files for inspection.

    Front3DDataset and room_done() both key on meta.json, so renaming it to
    meta.rejected.json removes the view from training without failing the
    room's other views or triggering a re-render on the next run.
    """
    meta_path = view_dir / 'meta.json'
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    meta['rejected'] = reason
    (view_dir / 'meta.rejected.json').write_text(json.dumps(meta, indent=2))
    meta_path.unlink(missing_ok=True)


def revalidate(out_root: Path, lightings: int) -> int:
    """Re-run validate_view over every view already on disk (no rendering).

    Previously quarantined views are restored first, so validator changes can
    both recover false positives and drop newly detected failures.
    """
    restored = kept = dropped = 0
    view_dirs = sorted({p.parent for p in out_root.rglob('meta*.json')})
    for vd in view_dirs:
        rej = vd / 'meta.rejected.json'
        if rej.is_file() and not (vd / 'meta.json').is_file():
            meta = json.loads(rej.read_text())
            meta.pop('rejected', None)
            (vd / 'meta.json').write_text(json.dumps(meta, indent=2))
            rej.unlink()
            restored += 1
        err = validate_view(vd, lightings)
        if err:
            quarantine_view(vd, err)
            dropped += 1
            print(f'[drop] {vd.relative_to(out_root)}: {err}')
        else:
            kept += 1
    print(f'Revalidated {kept + dropped} views: kept={kept} dropped={dropped} '
          f'(restored {restored} previously quarantined)')
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--glb-root', default=None, help='Extracted 3D-FRONT-TEST-SCENE tree (see module docstring); not needed with --revalidate')
    p.add_argument('--out', required=True)
    p.add_argument('--blender', default='tools/blender/blender-4.2.0-linux-x64/blender')
    p.add_argument('--views', type=int, default=3)
    p.add_argument('--lightings', type=int, default=2)
    p.add_argument('--resolution', type=int, default=512)
    p.add_argument('--samples', type=int, default=64)
    p.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--start', type=int, default=0, help='Room index slice start (for sharding)')
    p.add_argument('--end', type=int, default=None, help='Room index slice end (exclusive)')
    p.add_argument('--limit', type=int, default=None, help='Render at most N rooms this invocation')
    p.add_argument('--room-contains', default=None, help='Only rooms whose id contains this substring')
    p.add_argument('--revalidate', action='store_true',
                   help='No rendering: re-check all views on disk with the current validator, '
                        'restoring previously quarantined views first')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    if args.revalidate:
        return revalidate(out_root, args.lightings)
    if not args.glb_root:
        raise SystemExit('--glb-root is required unless --revalidate is given')
    glb_root = Path(args.glb_root).expanduser()
    blender = Path(args.blender)
    if not blender.is_file():
        raise FileNotFoundError(f'Blender not found at {blender} (pass --blender)')

    rooms = find_rooms(glb_root)
    if not rooms:
        raise FileNotFoundError(f'No *_full.glb under {glb_root} — extract the scene tar first (see docstring)')
    rooms = rooms[args.start:args.end]
    if args.room_contains:
        rooms = [r for r in rooms if args.room_contains in str(r)]

    blender_script = out_root / '_render_dataset_blender.py'
    blender_script.write_text(BLENDER_SCRIPT)
    manifest = out_root / 'manifest.jsonl'

    rendered = failed = skipped = 0
    for glb in rooms:
        if args.limit is not None and rendered >= args.limit:
            break
        room_id = room_id_of(glb, glb_root)
        room_out = out_root / room_id
        if room_done(room_out, args.views, args.lightings):
            skipped += 1
            continue
        # Per-room deterministic seed so shards and reruns reproduce.
        room_seed = args.seed * 1_000_003 + int(sha256(room_id.encode()).hexdigest()[:8], 16)
        cmd = [
            str(blender), '-b', '--python', str(blender_script), '--',
            '--glb', str(glb), '--out', str(room_out), '--room-id', room_id,
            '--views', str(args.views), '--lightings', str(args.lightings),
            '--resolution', str(args.resolution), '--samples', str(args.samples),
            '--seed', str(room_seed), '--device', args.device,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            status = f'blender_exit_{proc.returncode}'
        else:
            views = sorted(room_out.glob('view_*'))
            if not views:
                status = 'no_valid_view'
            else:
                kept, dropped = [], []
                for vd in views:
                    err = validate_view(vd, args.lightings)
                    if err:
                        quarantine_view(vd, err)
                        dropped.append(f'{vd.name}: {err}')
                    else:
                        kept.append(vd.name)
                if kept:
                    status = 'ok' if not dropped else f"ok ({len(kept)} views; dropped {'; '.join(dropped)})"
                else:
                    status = 'invalid: ' + '; '.join(dropped)
        with manifest.open('a') as f:
            f.write(json.dumps({'room_id': room_id, 'status': status, 'seed': room_seed}) + '\n')
        if status.startswith('ok'):
            rendered += 1
        else:
            failed += 1
            tail = '\n'.join(proc.stdout.splitlines()[-5:] + proc.stderr.splitlines()[-5:])
            print(f'[FAIL] {room_id}: {status}\n{tail}', file=sys.stderr)
        print(f'[{rendered} ok / {failed} fail / {skipped} skip] {room_id}: {status}', flush=True)

    print(f'Done. rendered={rendered} failed={failed} skipped={skipped} manifest={manifest}')
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())

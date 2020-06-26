import argparse
import math

from synth_ml.utils import rand
from synth_ml.blender import bl_utils, materials
from synth_ml.blender.wrappers import Scene, Object, Camera
import synth_ml.blender.callbacks as cb

deg = math.pi / 180

parser = argparse.ArgumentParser()
parser.add_argument("--frame-start", type=int, default=0)
parser.add_argument("--frame-end", type=int, default=1000)
args = parser.parse_args(bl_utils.get_user_args())

ts = cb.TransformSampler()

s = Scene.from_name('Scene')
cube = Object.from_name('Cube')
peg = Object.from_name('Peg')
cam = Object.from_name('Camera')
hole = Object.from_name('Hole')
hole_bottom = Object.from_name('HoleBottom')

material = materials.ProceduralNoiseVoronoiMix()
cube.assign_material(material)
hole_bottom_material = materials.ProceduralNoiseVoronoiMix()
hole_bottom.assign_material(hole_bottom_material)
# peg.assign_material(material)

ts.pos_sphere_volume(obj=cam, center=(0, .06, .12), radius=.005)
ts.rot_axis_uniform(cam, 'x', -23 * deg, -27 * deg)
ts.rot_axis_uniform(cam, 'y', -2 * deg, 2 * deg)
ts.rot_axis_uniform(cam, 'z', -3 * deg, 3 * deg)
ts.pos_axis_uniform(hole_bottom, 'z', -0.004, -.010)
ts.pos_axis_uniform(hole, 'z', 0, 0.01)

ts.pos_sphere_volume(obj=peg, center=(0, 0, .045), radius=.01)
for axis in 'xyz':
    ts.rot_axis_uniform(peg, axis, -10 * deg, 10 * deg)

# ts.scale_axis_uniform(cube, 'z', 0.001, 0.01)
# ts.scale_axis_uniform(hole, 'z', 0.001, 0.01)

mat_sampler_options = dict(
    scale=100, p_metallic=0.8,
    roughness=rand.NormalFloatSampler(0.1, .5, mi=0, ma=1),
    bevel_radius=rand.UniformFloatSampler(0.0005, 0.002)
)

s.callback = cb.CallbackCompose(
    ts,
    material.sampler_cb(**mat_sampler_options),
    hole_bottom_material.sampler_cb(**mat_sampler_options),
    cb.HdriEnvironment(scene=s, category='all', resolution=1, max_altitude_deg=90),
    cb.MetadataLogger(scene=s, objects=[peg]),
    cb.Renderer(
        scene=s, resolution=256, engines=('CYCLES',), denoise=True,
        cycles_samples=rand.NormalFloatSampler(mu=0, std=50, mi=5, ma=64, round=True),
    ),
)

s.render_frames(range(args.frame_start, args.frame_end))

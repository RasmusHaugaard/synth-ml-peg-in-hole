import argparse
import math

from synth_ml.utils import rand
from synth_ml.blender import bl_utils, materials
from synth_ml.blender.wrappers import Scene, Object
import synth_ml.blender.callbacks as cb


deg = math.pi / 180

parser = argparse.ArgumentParser()
parser.add_argument("--frame-start", type=int, default=0)
parser.add_argument("--frame-end", type=int, default=1000)
args = parser.parse_args(bl_utils.get_user_args())

ts = cb.TransformSampler()

s = Scene.from_name('Scene')
cube = Object.from_name('Cube')
peg_els = [Object.from_name('Peg_{}'.format(i)) for i in range(4)]
peg = Object.from_name('Peg')
cam = Object.from_name('Camera')
hole = Object.from_name('Hole')

materials = [materials.ProceduralNoiseVoronoiMix() for _ in range(4)]
cube.assign_material(materials[0])
for peg_el, mat_id in zip(peg_els, (1, 2, 2, 3)):
    peg_el.assign_material(materials[mat_id])

ts.pos_sphere_volume(obj=cam, radius=.01)
ts.rot_angle_axis(obj=cam, max_angle=5 * deg)
ts.pos_sphere_volume(obj=peg_els[0], radius=.01)
ts.rot_angle_axis(obj=peg_els[0], max_angle=10 * deg)
ts.rot_angle_axis(obj=peg_els[3], max_angle=30 * deg)

ts.scale_axis_uniform(cube, 'z', 0.001, 0.01)
ts.scale_axis_uniform(hole, 'z', 0.001, 0.01)

mat_sampler_options = dict(
    scale=100, p_metallic=0.8,
    roughness=rand.NormalFloatSampler(0.1, .5, mi=0, ma=1),
    bevel_radius=rand.UniformFloatSampler(0.0005, 0.002)
)

s.callback = cb.CallbackCompose(
    ts,
    *[m.sampler_cb(**mat_sampler_options) for m in materials],
    cb.HdriEnvironment(scene=s, category='all', resolution=1, max_altitude_deg=90),
    cb.MetadataLogger(scene=s, objects=[peg]),
    cb.Renderer(
        scene=s, resolution=256, engines=('CYCLES',), denoise=True,
        cycles_samples=rand.NormalFloatSampler(mu=0, std=50, mi=5, ma=64, round=True),
    ),
)

s.render_frames(range(args.frame_start, args.frame_end))

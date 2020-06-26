import matplotlib.pyplot as plt
import numpy as np
from synth_ml.blender.callbacks.metadata import Metadata
import vec_img_math

for i in range(10):
    img = plt.imread("synth_ml_data/cycles_denoise/{:04}.png".format(i))
    metadata = Metadata("synth_ml_data/metadata/{:04}.json".format(i))
    p_hole_image = metadata.world_2_image((0, 0, 0))
    p_peg_image = metadata.world_2_image(
        metadata.objects['Peg'].t_world @ np.array(((0, 0, -1, 1), (0, 0, 0, 1))).T
    )
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(img)
    axs[0].annotate("X", p_hole_image[:2], ha='center', va='center', c='w')
    for p in p_peg_image.T:
        axs[0].annotate("X", p[:2], ha='center', va='center', c='r')

    h, w = img.shape[:2]
    vec_img = vec_img_math.pos_to_vec_img(h, w, *p_hole_image[1::-1, 0])
    angle_img = vec_img_math.get_angle_img(vec_img)
    vec_img_math.plot_angle_img(angle_img, axs[1])

    vec_img = vec_img_math.line_to_vec_img(h, w, *p_peg_image[1::-1, 0], *p_peg_image[1::-1, 1])
    angle_img = vec_img_math.get_angle_img(vec_img)
    vec_img_math.plot_angle_img(angle_img, axs[2])

    plt.show()

import numpy as np
import matplotlib.pyplot as plt

from line_intersection import line_eq
from transform import Transform

deg = np.pi / 180


def get_cam_t_hole(hole_distance, azimuth, elevation):
    T = Transform(t=(0, -hole_distance, 0), rpy=(-np.pi / 2, 0, 0))
    T = Transform(rpy=(-elevation, 0, -azimuth)) @ T
    return T


def get_cam_K(resolution, fov):
    #  p_img = K @ p_cam
    #  u = x / z * alpha + u0
    assert 0 < fov < np.pi / 2
    assert 0 < resolution
    alpha = (resolution / 2) / np.tan(fov / 2)
    return np.array((
        (alpha, 0, resolution / 2),
        (0, alpha, resolution / 2),
        (0, 0, 1),
    ))


def cam_to_image(p_cam, K):
    p_img = K @ p_cam
    return p_img[:2] / p_img[2:]


def image_to_cam(p_img, K):
    return np.linalg.inv(K) @ np.concatenate((p_img, np.ones((1, 1))), axis=0)


def noise_transform(std_pos, max_angle):
    rotvec = np.random.randn(3)
    rotvec /= np.linalg.norm(rotvec)
    rotvec *= np.random.uniform(0, max_angle)
    t = np.random.randn(3) * std_pos
    return Transform(t=t, rotvec=rotvec)


def monte_carlo():
    elevation = 45 * deg
    cam_dist = 0.2
    cam_dist_xy = np.cos(elevation) * cam_dist * 0.5
    diameter = 0.01
    view_diameter_ratio = 10
    fov = np.arctan(diameter * view_diameter_ratio / 2 / cam_dist) * 2
    resolution = 200
    K = get_cam_K(resolution, fov)

    hole_world = np.array((0, 0, 0)).reshape(3, 1)
    peg_world = np.array((0, 0.005, 0.01)).reshape(3, 1)

    sigma_px = 1
    sigma_K = 2
    sigma_t_p = 0.01
    max_angle_t = 2 * deg

    N = 200
    alpha = .5 / N ** 0.1
    pt_size = 10

    print('fov', fov * 180 / np.pi)
    thetas = [np.pi / n for n in (2, 4, 8)]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_xlim(0, resolution)
    ax.set_ylim(0, resolution)
    center = (resolution / 2, resolution / 2)
    ax.add_artist(plt.Circle(center, resolution / view_diameter_ratio / 2))
    ax.scatter(*(sigma_px * np.random.randn(N, 2) + center).T, s=pt_size, alpha=alpha, lw=0, c='r', zorder=3)
    ax.set_aspect(1)
    ax.set_xlabel('u [px]')
    ax.set_ylabel('v [px]')
    plt.tight_layout()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    text = 'elevation: {:.2f} deg'.format(elevation / deg)
    text += ', c2w unc.: ({:.2f} deg, {:.2f} mm)'.format(max_angle_t / deg, sigma_t_p * 1e3)
    text += ', vision unc.: {:.2f}'.format(sigma_px)
    text += ', K unc.: {:.2f}'.format(sigma_K)
    fig.suptitle(text)

    for i, (theta, ax) in enumerate(zip(thetas, axs.reshape(-1))):
        cams_t_world = [get_cam_t_hole(cam_dist, azimuth, elevation) for azimuth in (0, theta)]

        holes = []
        pegs = []
        for _ in range(N):
            Ks_ = K.copy(), K.copy()
            for K_ in Ks_:
                K_[:2] += np.random.randn(2, 3) * sigma_K
            c2ws_ = [c2w @ noise_transform(sigma_t_p, max_angle_t) for c2w in cams_t_world]
            for p, ls in zip((hole_world, peg_world), (holes, pegs)):
                p_cams = [c2w.inv() @ p for c2w in c2ws_]
                p_imgs_ = [cam_to_image(p_cam, K) + np.random.randn(2, 1) * sigma_px for p_cam in p_cams]
                p_cams_ = [image_to_cam(p_img_, K_) for p_img_, K_ in zip(p_imgs_, Ks_)]
                p_lines = [line_eq((c2w @ p_cam_)[:2, 0], c2w.t[:2]) for c2w, p_cam_ in
                           zip(cams_t_world, p_cams_)]
                p_ = np.cross(*p_lines)
                p_ = p_[:2] / p_[2]
                ls.append(p_)
        holes = np.array(holes)
        pegs = np.array(pegs)

        for ps, name in zip((holes, pegs, pegs - holes), ('hole', 'peg', 'direction')):
            ax.scatter(*ps.T, alpha=alpha, s=pt_size, lw=0, label=name)
        # plt.hexbin(*ps.T, extent=(-hole_distance, hole_distance, -hole_distance, hole_distance), cmap=c, alpha=0.3)
        a_t, b_t = cams_t_world[0].t[:2], cams_t_world[1].t[:2]
        ax.scatter([a_t[0], b_t[0]], [a_t[1], b_t[1]], label='cams')
        ax.grid()
        ax.add_artist(plt.Circle(hole_world[:2, 0], diameter / 2, fill=False, color='blue'))
        ax.add_artist(plt.Circle(peg_world[:2, 0], diameter / 2, fill=False, color='darkorange'))
        ax.set_aspect(1)
        ax.set_xlim(-cam_dist_xy, cam_dist_xy)
        ax.set_ylim(-cam_dist_xy, cam_dist_xy)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('azimuth diff: {:.2f} deg'.format(theta / deg))
        ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    monte_carlo()

import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from scipy.ndimage import filters
from typing import Union


def get_px_pos_img(h, w):
    px_pos_img = np.zeros((h, w, 2))
    px_pos_img[:, :, 0] = np.arange(h).reshape(h, 1)
    px_pos_img[:, :, 1] = np.arange(w).reshape(1, w)
    return px_pos_img


def pos_to_vec_img(h, w, y, x):
    vector_img = get_px_pos_img(h, w)
    vector_img[:, :, 0] -= y
    vector_img[:, :, 1] -= x
    vector_img /= np.linalg.norm(vector_img, axis=2, keepdims=True)
    return vector_img


def line_to_vec_img(h, w, y0, x0, y1, x1):
    # (a*x + b*y = c)
    a, b, c = points_to_line(x0, y0, x1, y1)
    px_pos_img = get_px_pos_img(h, w)
    line_image = a * px_pos_img[:, :, 1] + b * px_pos_img[:, :, 0]
    mask = line_image < c
    v = np.array((b, a)) / np.linalg.norm((b, a))
    vector_img = np.empty((h, w, 2))
    vector_img[:, :] = v
    vector_img[mask] = -v
    return vector_img


def get_angle_img(vector_img):
    return np.arctan2(vector_img[:, :, 0], vector_img[:, :, 1])


def plot_angle_img(angle_img, ax=None):
    ax = ax or plt
    if len(angle_img.shape) == 3:
        angle_img = get_angle_img(angle_img)
    ax.imshow(angle_img, cmap='hsv', vmin=-math.pi, vmax=math.pi)


def norm_grad_from_vec_img(vector_field: np.ndarray, ksize=3):
    #vector_field = vector_field / np.linalg.norm(vector_field, axis=-1, keepdims=True)  # TODO: possibly remove again
    sobel_x = cv2.Sobel(vector_field[:, :, 1], cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(vector_field[:, :, 0], cv2.CV_64F, 0, 1, ksize=ksize)
    sobel_xy = (sobel_x + sobel_y) / 8 + 0.5
    return sobel_xy


def points_from_grad_img(grad_img, threshold):
    h, w = grad_img.shape[:2]
    hole_max_area = filters.maximum_filter(grad_img, size=5)
    mask = np.logical_and(hole_max_area == grad_img, grad_img > threshold)
    points = np.argwhere(mask)
    c_dist = np.linalg.norm(points - (h / 2, w / 2), axis=1)
    c_dist_sort_idx = np.argsort(c_dist)
    points = points[c_dist_sort_idx]
    return points


def edges_from_grad_img(grad_img, threshold=0.8, dilate=True):
    edges = (grad_img > threshold).astype(np.uint8) * 255
    if dilate:
        edges = cv2.dilate(edges, np.ones((3, 3)))
    return edges


def hough_lines(edges, rho_res=2, theta_res=np.pi / 180 * 0.5, min_votes=200, min_length=250, max_line_gap=50):
    lines = cv2.HoughLinesP(edges, rho_res, theta_res, min_votes, min_length, max_line_gap)
    if lines is None:
        lines = np.array([])
    return lines.reshape((-1, 4))


def points_to_line(x0, y0, x1, y1):
    a = y1 - y0
    b = x0 - x1
    c = a * x0 + b * y0
    return a, b, c


COLORS = {
    'r': (255, 0, 0, 255),
    'g': (0, 255, 0, 255),
    'b': (0, 0, 255, 255),
    'k': (0, 0, 0, 255),
    'w': (255, 255, 255, 0),
}


def draw_points(img, points, c: Union[str, tuple] = 'r'):
    if isinstance(c, str):
        c = COLORS[c]
    for i, p in enumerate(points):
        cv2.drawMarker(img, tuple(p[::-1]), c, cv2.MARKER_TILTED_CROSS, 10, 1, cv2.LINE_AA)


def draw_lines_from_point_pairs(img, point_pairs, extend=False, c: Union[str, tuple] = 'g'):
    _c = c
    if isinstance(_c, str):
        _c = COLORS[_c]
    h, w = img.shape[:2]
    for x0, y0, x1, y1 in point_pairs:
        if extend:
            # (a*x + b*y = c) => (x = (c - b*y)/a)
            a, b, c = points_to_line(x0, y0, x1, y1)
            y0, y1 = 0, h
            x0 = int(round((c - b * y0) / a))
            x1 = int(round((c - b * y1) / a))
        cv2.line(img, (x0, y0), (x1, y1), _c, 2, cv2.LINE_AA)

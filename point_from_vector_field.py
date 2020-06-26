import numpy as np
import cv2


def point_from_vector_field(vector_field: np.ndarray, ksize=3):
    sobel_x = cv2.Sobel(vector_field[:, :, 0], cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(vector_field[:, :, 1], cv2.CV_64F, 0, 1, ksize=ksize)
    sobel_xy = np.maximum(sobel_x + sobel_y, 0)
    point = np.unravel_index(np.argmax(sobel_xy), sobel_xy.shape)
    return point, sobel_xy


def _main():
    from dataset import PegInHoleDataset
    import matplotlib.pyplot as plt

    data = PegInHoleDataset("synth_ml_data")
    img, vector_fields = data.get(0)

    point, sobel_xy = point_from_vector_field(vector_fields[0])

    plt.imshow(sobel_xy)
    print(point)
    plt.annotate("X", point[::-1], ha="center", va="center")

    plt.show()
    plt.imshow(img)
    plt.annotate("X", point[::-1], ha="center", va="center")
    plt.show()


if __name__ == '__main__':
    _main()

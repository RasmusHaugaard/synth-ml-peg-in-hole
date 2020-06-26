from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from fastai.vision import imagenet_stats

from synth_ml.blender.callbacks.metadata import Metadata

import vec_img_math


class PegInHoleDataset(torch.utils.data.Dataset):
    empty_val = False

    def __init__(self, data_root, valid=False, representation='vector',
                 sigma=3,  # for heatmap rep
                 peg_line=True,  # for vector rep
                 coco_composite=False,
                 rand_crop=False, flip_horizontal=False, N=1000):
        self.data_root = Path(data_root)
        self.valid = valid
        self.sigma = sigma
        self.peg_line = peg_line
        self.representation = representation
        self.rand_crop = rand_crop
        self.coco_composite = coco_composite
        self.flip_horizontal = flip_horizontal
        self.N = N
        self.N_VALID = N // 10
        self.N_TRAIN = N - self.N_VALID
        if representation == 'vector':
            self.c = 4
        elif representation == 'heatmap':
            self.c = 2
        else:
            raise ValueError('unknown representation "{}"'.format(representation))

    def __len__(self):
        return self.N_VALID if self.valid else self.N_TRAIN

    def get_rep(self, idx):
        data_path = self.data_root / 'metadata' / '{:04}.json'.format(idx)
        metadata = Metadata(data_path)

        w, h = metadata.resolution
        p_hole_img = metadata.world_2_image((0, 0, 0))
        p_peg_img = metadata.world_2_image(
            metadata.objects['Peg'].t_world @ np.array(((0, 0, -1, 1), (0, 0, 0, 1))).T
        )

        if self.representation == 'vector':
            vector_fields = np.empty((2, h, w, 2))
            vector_fields[0] = vec_img_math.pos_to_vec_img(h, w, *p_hole_img[1::-1, 0])
            if self.peg_line:
                vector_fields[1] = vec_img_math.line_to_vec_img(h, w, *p_peg_img[1::-1, 0], *p_peg_img[1::-1, 1])
            else:
                vector_fields[1] = vec_img_math.pos_to_vec_img(h, w, *p_peg_img[1::-1, 0])
            return vector_fields
        elif self.representation == 'heatmap':
            heatmaps = np.empty((2, h, w))
            heatmaps[0] = heatmap(self.sigma, w, h, p_hole_img[:2].T)
            heatmaps[1] = heatmap(self.sigma, w, h, p_peg_img[:2, 0:1].T)
            return heatmaps
        else:
            raise ValueError('unknown representation "{}"'.format(self.representation))

    def get(self, idx):
        if self.valid:
            idx = idx + self.N_TRAIN
        img_path = self.data_root / 'cycles_denoise' / '{:04}.png'.format(idx)
        img = Image.open(str(img_path))
        if self.coco_composite:
            img = coco_composite(img)[0]
        rep = self.get_rep(idx)
        if self.rand_crop:
            img = np.array(img)
            h, w = img.shape[:2]
            crop_size = 224
            crop_start = np.random.rand(2) * (h - crop_size, w - crop_size)
            h0, w0 = np.round(crop_start).astype(int)
            img = img[h0:h0 + crop_size, w0:w0 + crop_size]
            rep = rep[:, h0:h0 + crop_size, w0:w0 + crop_size]
        if self.flip_horizontal and np.random.rand() < 0.5:
            img = img[:, ::-1].copy()
            rep = rep[:, :, ::-1].copy()
            if self.representation == 'vector':
                rep[..., 1] *= -1
        if np.random.rand() < 0.5:
            k = np.random.randint(1, 3) * 2 + 1
            img = cv2.blur(img, (k, k))
        return img, rep

    def normalize(self, img, rep):
        img = torchvision.transforms.ToTensor()(img)
        img = torchvision.transforms.Normalize(*imagenet_stats)(img)

        if self.representation == 'vector':
            rep = rep.transpose((0, 3, 1, 2))
            rep = rep.reshape((-1, *rep.shape[2:]))
        rep = torch.from_numpy(rep).float()

        return img, rep

    def __getitem__(self, idx):
        return self.normalize(*self.get(idx))


def heatmap(sigma, w, h, points, d=3):  # efficient version of heatmap naive
    s = int(sigma * d)  # assumes that values further away than sigma * d are insignificant
    hm = np.zeros((h, w))
    for x, y in points:
        _x, _y = int(round(x)), int(round(y))
        xmi, xma = max(0, _x - s), min(w, _x + s)
        ymi, yma = max(0, _y - s), min(h, _y + s)
        _h, _w = yma - ymi, xma - xmi
        X, Y = np.arange(_w).reshape(1, _w), np.arange(_h).reshape(_h, 1)
        _hm = (x - xmi - X) ** 2 + (y - ymi - Y) ** 2
        _hm = np.exp(-_hm / (2 * sigma ** 2))
        hm[ymi:yma, xmi:xma] = np.maximum(hm[ymi:yma, xmi:xma], _hm)
    return hm


coco_imgs = list(Path('/home/rlha/data/coco_val').glob('*.jpg'))


def load_rand_coco() -> Image.Image:
    fp = np.random.choice(coco_imgs)
    return Image.open(fp)


def coco_composite(img: Image):
    overlay = load_rand_coco().convert('RGB').resize((img.width, img.height), Image.BILINEAR)
    mask = load_rand_coco().convert('L').resize((img.width, img.height), Image.BILINEAR)
    mask = ImageEnhance.Brightness(mask).enhance(.75)
    return Image.composite(overlay, img, mask), overlay, mask


def main():
    dataset = PegInHoleDataset('synth_ml_data', representation='heatmap', coco_composite=True)
    img, rep = dataset.get(0)
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.imshow(rep[0])
    # vec_img_math.plot_angle_img(vec_img_math.get_angle_img(vector_fields[0]))
    plt.subplot(1, 3, 3)
    plt.imshow(rep[1])
    # vec_img_math.plot_angle_img(vec_img_math.get_angle_img(vector_fields[1]))
    plt.show()
    print(dataset[0][1].shape)


if __name__ == '__main__':
    main()

import numpy as np
import cv2
import torch
import torchvision
from ml.models.unet import ResNetUNet
import fastai.vision as faiv
from point_from_vector_field import point_from_vector_field
from dataset import PegInHoleDataset
import matplotlib.pyplot as plt
from scipy.ndimage import filters

cap = cv2.VideoCapture("real.mp4")
model = ResNetUNet(4)
model.load_state_dict(torch.load("models/model.pth"))
model.cuda(1)
model.eval()
torch.no_grad()

hole_thresh = 2

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    h, w = frame.shape[:2]
    h_start = (h - w) // 2
    frame = frame[h_start:h_start + w]
    frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _img = torchvision.transforms.ToTensor()(rgb)
    _img = torchvision.transforms.Normalize(*faiv.imagenet_stats)(_img)
    result = model(_img.view(1, *_img.shape).cuda(1))
    result = result.view(result.shape[1:]).detach().cpu().numpy()
    result = result.reshape((2, 2, *result.shape[1:]))
    result = result.transpose((0, 2, 3, 1))

    hole, hole_sobel_xy = point_from_vector_field(result[0])
    peg, peg_sobel_xy = point_from_vector_field(result[1])
    angle_img_hole = PegInHoleDataset.get_angle_image(result[0])
    angle_img_peg = PegInHoleDataset.get_angle_image(result[0])
    hole_max_area = filters.maximum_filter(hole_sobel_xy, size=5)
    mask = np.logical_and(hole_max_area == hole_sobel_xy, hole_sobel_xy > 2)
    holes = np.argwhere(mask)
    for hole in holes:
        cv2.drawMarker(frame, (hole[1], hole[0]), (255, 255, 255), cv2.MARKER_CROSS, markerSize=10, thickness=1)
    cv2.drawMarker(frame, (peg[1], peg[0]), (255, 0, 0), cv2.MARKER_CROSS, markerSize=10, thickness=1)
    cv2.imshow('frame', frame)
    cv2.imshow('hole sobel', hole_sobel_xy / 4)
    cv2.imshow('peg sobel', peg_sobel_xy / 4)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('p'):
        cv2.waitKey()
    if key & 0xFF == ord('c'):
        plt.imshow(angle_img_hole, cmap='hsv')
        plt.show()

cap.release()
cv2.destroyAllWindows()

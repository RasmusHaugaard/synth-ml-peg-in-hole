import argparse
from functools import partial
import queue

import cv2
import numpy as np
import torch
import torchvision
import fastai.vision as faiv
from PIL import Image

import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from sensor_msgs.msg import Image as ImageMsg
from ros_numpy.image import numpy_to_image, image_to_numpy

from vec_img_math import draw_points
from ml.models.unet import ResNetUNet

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--model', default='models/synth-e30-lr1e-3-wd1e-4-c0.75.pth')
args = parser.parse_args()

device = torch.device(args.device)
model = ResNetUNet(2, pretrained=False).half().to(device)
model.load_state_dict(
    torch.load(args.model, map_location=lambda s, l: s)['model']
)


def infer(model, img: Image) -> np.ndarray:
    img = torchvision.transforms.ToTensor()(img)
    img = torchvision.transforms.Normalize(*faiv.imagenet_stats)(img)
    with torch.no_grad():
        result = model(img.half().unsqueeze(0).to(device)).squeeze(0)
    return result.detach().cpu().numpy()


def main():
    rospy.init_node('annotator')

    annotation_pubs = []
    annotated_img_pubs = []
    in_qs = []

    def cb(img_msg: ImageMsg, q: queue.Queue):
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        finally:
            q.put(img_msg)

    for i, img_suffix in enumerate('ab'):
        annotated_img_pub = rospy.Publisher('img_annotated_{}'.format(img_suffix), ImageMsg, queue_size=1)
        annotated_img_pubs.append(annotated_img_pub)
        annotation_pub = rospy.Publisher('points_{}'.format(img_suffix), numpy_msg(Floats), queue_size=1)
        annotation_pubs.append(annotation_pub)
        in_qs.append(queue.Queue(maxsize=1))
        rospy.Subscriber("img_{}".format(img_suffix), ImageMsg, partial(cb, q=in_qs[i]), queue_size=1)

    while not rospy.is_shutdown():
        for inq, annotation_pub, annotated_img_pub, in zip(in_qs, annotation_pubs, annotated_img_pubs):
            # sequential inference in the main thread, interlaced between image sources
            try:
                img = image_to_numpy(inq.get(timeout=1.))
            except queue.Empty:
                continue
            hms = infer(model, Image.fromarray(img))
            points = []
            for hm in hms:
                points.append(np.unravel_index(np.argmax(hm), hm.shape))
            annotation_pub.publish(np.array(points).astype(np.float32).reshape(-1))
            for p, c in zip(points, 'br'):
                draw_points(img, [p], c=c)
            annotated_img_pub.publish(numpy_to_image(img, 'rgb8'))


if __name__ == '__main__':
    main()

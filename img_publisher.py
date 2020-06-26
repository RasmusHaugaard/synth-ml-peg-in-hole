from pathlib import Path
import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image as ImageMsg
from ros_numpy.image import numpy_to_image, image_to_numpy

image_fps = list(Path('/home/rlha/d/table-calib/data/dataset_0/').glob('*.png'))

pub_a = rospy.Publisher('img_a', ImageMsg, queue_size=1)
pub_b = rospy.Publisher('img_b', ImageMsg, queue_size=1)
rospy.init_node('img_publisher')
rate = rospy.Rate(1)  # Hz

s, u, l, e = 500, 350, 700, 70
while not rospy.is_shutdown():
    img = cv2.imread(str(np.random.choice(image_fps)))
    e_ = np.random.randint(-e, e, 2)
    u_, l_ = u + e_[0], l + e_[1]
    img = img[u_:u_+s, l_:l_+s]
    img = cv2.resize(img, (224, 224))
    for pub in pub_a, pub_b:
        pub.publish(numpy_to_image(img[..., ::-1], 'rgb8'))
    rate.sleep()

from detector import ATSSHumanFaceAssocDetector, draw_associations
from matplotlib import pyplot as plt
import cv2
import numpy as np
import sys
cv2.setNumThreads(min(4, cv2.getNumberOfCPUs()))
np.set_printoptions(formatter={'float': '{:.5f}'.format})

if __name__ == '__main__':
    img = cv2.imread('example.png')
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    scale = 512 / np.max(img.shape)
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    detector = ATSSHumanFaceAssocDetector(use_gpu=False, fp16=False)
    human_boxes, face_boxes, association = detector.predict(img)

    img = draw_associations(img, human_boxes, face_boxes, association)
    cv2.imwrite('example_vis.png', img[..., ::-1])
    #plt.imshow(img)
    #plt.show()




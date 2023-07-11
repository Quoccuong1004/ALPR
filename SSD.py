import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import torchvision.transforms as transforms
import functools
from model import create_model

from config import (
    NUM_CLASSES, DEVICE, CLASSES
)

from xml.etree import ElementTree as et
from config import PIC_DIR, ANNO_DIR
import time
from tqdm import tqdm

def compare(rect1, rect2):
        if abs(rect1[1] - rect2[1]) > 40:
            return rect1[1] - rect2[1]
        else:
            return rect1[0] - rect2[0]

def find_index(Box_sorted, box):
    index_value = []
    for i in Box_sorted:
        index = 0
        for k in box:
            if k[0] == i[0] and k[1] == i[1] and k[2] == i[2] and k[3] == i[3] :
                index_value.append(index)
                break
            else:
                index += 1
    return index_value

def SSD300(image, model, detection_threshold):
    size = 300

    if image.shape[1]/image.shape[0] > 2.3:
        image = cv2.resize(image, (3*size, size))
    else:
        image = cv2.resize(image, (size, size))

    orig_image = image.copy()
    # print(image.shape[1], image.shape[0])

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image, dtype=torch.float)

    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image.to(DEVICE))
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    plate = ""
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        Boxes = sorted(boxes, key=functools.cmp_to_key(compare))
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        index = find_index(Boxes, boxes)
        for i in index:
            plate += pred_classes[i]
        # COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        # for j, box in enumerate(boxes):
        #     class_name = pred_classes[j]
        #     color = COLORS[CLASSES.index(class_name)]
        #     if orig_image.shape[1]/orig_image.shape[0] > 2.3 :
        #         cv2.rectangle(orig_image,
        #                     (int(box[0]), 3*int(box[1])),
        #                     (int(box[2]), 3*int(box[3])),
        #                     color, 2)
        #         cv2.putText(orig_image, class_name,
        #                     (int(box[0]), 3*int(box[1]-5)),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,
        #                     2, lineType=cv2.LINE_AA)
        #     else:
        #         cv2.rectangle(orig_image,
        #                 (int(box[0]), int(box[1])),
        #                 (int(box[2]), int(box[3])),
        #                 color, 2)
        #         cv2.putText(orig_image, class_name,
        #                 (int(box[0]), int(box[1]-5)),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,
        #                 2, lineType=cv2.LINE_AA)
    return plate
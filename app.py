import subprocess
import tempfile
import time
from pathlib import Path
import cv2
import gradio as gr
import sys
import torch
sys.path.insert(0, "./faster_RCNN")
from fasterRCNN import faster_RCNN
from inferer import Inferer
from model import create_model, create_model_ssd
from config import (
    NUM_CLASSES, DEVICE
)

from CNN import LeNet, CNN_model
from SSD import SSD300

pipeline = Inferer("weights/yolo_license_plate.pt", "gpu", "data/mydataset.yaml", 640)
#load fasterRCNN weight
detection_threshold = 0.80
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('./weights/faster_rcnn.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

#load lenet
model_lenet = LeNet()
model_lenet.load_state_dict(torch.load('./weights/lenet.pth',map_location=DEVICE))
model_lenet.to(DEVICE).eval()

#load ssd
detection_threshold_ssd = 0.2
model_ssd = create_model_ssd(num_classes=NUM_CLASSES, size=300)
checkpoint_ssd = torch.load('./weights/ssd.pth', map_location=DEVICE)
model_ssd.load_state_dict(checkpoint_ssd['model_state_dict'])
model_ssd.to(DEVICE).eval()

print(f"GPU on? {'🟢' if pipeline.device.type != 'cpu' else '🔴'}")

def fn_image(image, models,conf_thres, iou_thres):
    result_1 = pipeline(image, conf_thres, iou_thres)
    if result_1 is None:
        return
    else:
        predict_yolo, bbox_cut, xyxy = result_1
        for i in range(len(bbox_cut)):
            if models == 'fasterRCNN':
                result = faster_RCNN(bbox_cut[i], detection_threshold=detection_threshold, model=model)
            elif models == 'LeNet':
                result = CNN_model(bbox_cut[i], model_lenet, DEVICE)
            elif models == 'SSD':
                result = SSD300(bbox_cut[i], detection_threshold=detection_threshold_ssd, model=model_ssd)
            lw = max(round(sum(image.shape) / 2 * 0.003), 2)
            # Get the coordinates of the text position
            text_x, text_y = int(xyxy[i][0].item()), int(xyxy[i][1].item())

            # Measure the width and height of the text
            (text_width, text_height), _ = cv2.getTextSize(text=result, fontFace=0, fontScale=lw/2, thickness=3)

            # Calculate the coordinates of the rectangle
            rect_x = text_x
            rect_y = text_y - text_height - 2
            rect_x2 = text_x + text_width
            rect_y2 = text_y

            # Draw the white rectangle
            cv2.rectangle(img=predict_yolo, pt1=(rect_x, rect_y), pt2=(rect_x2, rect_y2), color=(255, 255, 255), thickness=cv2.FILLED)

            
            # Draw the text on the rectangle
            cv2.putText(img=predict_yolo, text=result, org=(text_x, text_y-2), fontFace=0, fontScale=lw/2,
                        color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
    return predict_yolo


image_interface = gr.Interface(
    fn=fn_image,
    inputs=[
        "image",
        gr.Dropdown(['fasterRCNN', 'LeNet', 'SSD'], label="Models"),
        gr.Slider(0, 1, value=0.5, label="Confidence Threshold"),
        gr.Slider(0, 1, value=0.5, label="IOU Threshold"),
    ],
    outputs=gr.Image(type="filepath"),
 #   examples=[["example_1.jpg", 0.5, 0.5], ["example_2.jpg", 0.25, 0.45], ["example_3.jpg", 0.25, 0.45]],
    title="ALPR",
    description=(
        "Gradio demo for Automatic License Plate Recognition."
    ),
    allow_flagging=False,
    allow_screenshot=False,
)

if __name__ == "__main__":
    gr.TabbedInterface(
        [image_interface],
        ["Run on Images!"],
    ).launch(share = True)

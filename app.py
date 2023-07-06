import subprocess
import tempfile
import time
from pathlib import Path
import cv2
import gradio as gr
import sys
sys.path.insert(0, "./faster_RCNN")
from fasterRCNN import faster_RCNN
from inferer import Inferer

pipeline = Inferer("weights/yolo_license_plate.pt", "cpu", "data/mydataset.yaml", 640)
print(f"GPU on? {'🟢' if pipeline.device.type != 'cpu' else '🔴'}")

def fn_image(image, conf_thres, iou_thres):
    result_1 = pipeline(image, conf_thres, iou_thres)
    
    if result_1 is None:
        return
    else:
        predict_yolo, bbox_cut, xyxy = result_1
        result = faster_RCNN(bbox_cut)
        lw = max(round(sum(image.shape) / 2 * 0.003), 2)
        # Get the coordinates of the text position
        text_x, text_y = int(xyxy[0].item()), int(xyxy[1].item())

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


def fn_video(video_file, conf_thres, iou_thres, start_sec, duration):
    start_timestamp = time.strftime("%H:%M:%S", time.gmtime(start_sec))
    end_timestamp = time.strftime("%H:%M:%S", time.gmtime(start_sec + duration))

    suffix = Path(video_file).suffix

    clip_temp_file = tempfile.NamedTemporaryFile(suffix=suffix)
    subprocess.call(
        f"ffmpeg -y -ss {start_timestamp} -i {video_file} -to {end_timestamp} -c copy {clip_temp_file.name}".split()
    )

    # Reader of clip file
    cap = cv2.VideoCapture(clip_temp_file.name)

    # This is an intermediary temp file where we'll write the video to
    # Unfortunately, gradio doesn't play too nice with videos rn so we have to do some hackiness
    # with ffmpeg at the end of the function here.
    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_file:
        out = cv2.VideoWriter(temp_file.name, cv2.VideoWriter_fourcc(*"MP4V"), 30, (1280, 720))

        num_frames = 0
        max_frames = duration * 30
        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    break
            except Exception as e:
                print(e)
                continue
            print("FRAME DTYPE", type(frame))
            out.write(pipeline(frame, conf_thres, iou_thres))
            num_frames += 1
            print("Processed {} frames".format(num_frames))
            if num_frames == max_frames:
                break

        out.release()

        # Aforementioned hackiness
        out_file = tempfile.NamedTemporaryFile(suffix="out.mp4", delete=False)
        subprocess.run(f"ffmpeg -y -loglevel quiet -stats -i {temp_file.name} -c:v libx264 {out_file.name}".split())

    return out_file.name


image_interface = gr.Interface(
    fn=fn_image,
    inputs=[
        "image",
        gr.Slider(0, 1, value=0.5, label="Confidence Threshold"),
        gr.Slider(0, 1, value=0.5, label="IOU Threshold"),
    ],
    outputs=gr.Image(type="filepath"),
 #   examples=[["example_1.jpg", 0.5, 0.5], ["example_2.jpg", 0.25, 0.45], ["example_3.jpg", 0.25, 0.45]],
    title="YOLOv6",
    description=(
        "Gradio demo for YOLOv6 for object detection on images. To use it, simply upload your image or click one of the"
        " examples to load them. Read more at the links below."
    ),
    article=(
        "<div style='text-align: center;'><a href='https://github.com/meituan/YOLOv6' target='_blank'>Github Repo</a>"
        " <center><img src='https://visitor-badge.glitch.me/badge?page_id=nateraw_yolov6' alt='visitor"
        " badge'></center></div>"
    ),
    allow_flagging=False,
    allow_screenshot=False,
)

video_interface = gr.Interface(
    fn=fn_video,
    inputs=[
        gr.Video(type="file"),
        gr.Slider(0, 1, value=0.25, label="Confidence Threshold"),
        gr.Slider(0, 1, value=0.45, label="IOU Threshold"),
        gr.Slider(0, 10, value=0, label="Start Second", step=1),
        gr.Slider(0, 10 if pipeline.device.type != 'cpu' else 3, value=4, label="Duration", step=1),
    ],
    outputs=gr.Video(type="filepath", format="mp4"),
    # examples=[
    #     ["example_1.mp4", 0.25, 0.45, 0, 2],
    #     ["example_2.mp4", 0.25, 0.45, 5, 3],
    #     ["example_3.mp4", 0.25, 0.45, 6, 3],
    # ],
    title="YOLOv6",
    description=(
        "Gradio demo for YOLOv6 for object detection on videos. To use it, simply upload your video or click one of the"
        " examples to load them. Read more at the links below."
    ),
    article=(
        "<div style='text-align: center;'><a href='https://github.com/meituan/YOLOv6' target='_blank'>Github Repo</a>"
        " <center><img src='https://visitor-badge.glitch.me/badge?page_id=nateraw_yolov6' alt='visitor"
        " badge'></center></div>"
    ),
    allow_flagging=False,
    allow_screenshot=False,
)

webcam_interface = gr.Interface(
    fn_image,
    inputs=[
        gr.Image(source='webcam', streaming=True),
        gr.Slider(0, 1, value=0.5, label="Confidence Threshold"),
        gr.Slider(0, 1, value=0.5, label="IOU Threshold"),
    ],
    outputs=gr.Image(type="filepath"),
    live=True,
    title="YOLOv6",
    description=(
        "Gradio demo for YOLOv6 for object detection on real time webcam. To use it, simply allow the browser to access"
        " your webcam. Read more at the links below."
    ),
    article=(
        "<div style='text-align: center;'><a href='https://github.com/meituan/YOLOv6' target='_blank'>Github Repo</a>"
        " <center><img src='https://visitor-badge.glitch.me/badge?page_id=nateraw_yolov6' alt='visitor"
        " badge'></center></div>"
    ),
    allow_flagging=False,
    allow_screenshot=False,
)

if __name__ == "__main__":
    gr.TabbedInterface(
        [video_interface, image_interface, webcam_interface],
        ["Run on Videos!", "Run on Images!", "Run on Webcam!"],
    ).launch(share = True)

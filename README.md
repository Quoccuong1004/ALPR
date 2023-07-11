## ALPR - Automatic License Plate Recognition 
### Description
A system that could accurately identify and recognize the characters on a license plate from an image
### Technologies used:
## License Plate Detection
- YOLOv6
## Character Recognition
- Single Shot Detection (one-stage)
- Segmentation and using CNN to recognize
- Faster R-CNN (two-stage)

### Installation
**How to start the project:**
```
>>> Clone the repository and change it on the command line:
git clone https://github.com/Quoccuong1004/ALPR.git
cd ALPR
```
```
>>> Install dependencies from requirements.txt:
python -m pip install --upgrade pip (MacOS python3 -m pip install --upgrade pip for MacOS)
pip install -r requirements.txt (python3 pip install -r requirements.txt for MacOS)
```
```
>>> Install gradio:
pip install gradio (python3 pip install gradio for MacOS)
```
>>><strong>Important - Download the weights of models</strong>
- Open this <a href="https://drive.google.com/drive/folders/1mlu-t7ZW3XIF43dmByD7oZPmipYQ4uNY?fbclid=IwAR12tb-a4CetJj1IyzqLGMlxNR7tGC3qIwrVI_LTzUXV8VPHDzEsVKrsXeQ">Google Drive</a> to download
- Download all the weights of models
- Create a 'weights' folder in the ALPR project
- Save them in ALPR/weights folder
```
>>> Run project:
python app.py (if you are a MacOS user python3 app.py)
```

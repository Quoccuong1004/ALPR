import torch

BATCH_SIZE = 2 # increase / decrease according to GPU memeory
WIDTH = 300
HEIGHT_LONG = 75
HEIGHT_SHORT = 215
NUM_EPOCHS = 10 # number of epochs to train for
NUM_WORKERS = 0

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML files directory
PIC_DIR = '/content/drive/MyDrive/ALPR/data/voc_plate_ocr_dataset/ocr_az_dataset'
ANNO_DIR = '/content/drive/MyDrive/ALPR/data/voc_plate_ocr_dataset/Annotations'
TRAIN_PART = '/content/drive/MyDrive/ALPR/data/voc_plate_ocr_dataset/ImageSets/Main/train.txt'

# validation images and XML files directory
VALID_PART = '/content/drive/MyDrive/ALPR/data/voc_plate_ocr_dataset/ImageSets/Main/val.txt'

# classes: 0 index is reserved for background
CLASSES = ["__background__","0","1","2","3","4","5","6","7","8","9",
           "A","B","C","D","E","F","G","H","I","J","K","L",
           "M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]


NUM_CLASSES = len(CLASSES)

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = True

# location to save model and plots
OUT_DIR = '/content/drive/MyDrive/ALPR/src/faster_RCNN/weights'
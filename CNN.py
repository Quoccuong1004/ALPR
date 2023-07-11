import cv2

import functools

import numpy as np

import torch

import torch.nn.functional as F

from torchvision import transforms

import torch.nn as nn




classes = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H"

,"K","L","M","N","P","R","S","T","U","V","X","Y","Z"]

num_classes = 31




class LeNet(nn.Module):

     def __init__(self):

         super(LeNet, self).__init__()




         # define the layers of the model

         self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)

         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

         self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

         self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)

         self.fc2 = nn.Linear(in_features=120, out_features=84)

         self.fc3 = nn.Linear(in_features=84, out_features=num_classes)




     def forward(self, x):

         # apply the layers in the model to the input x

         x = self.pool1(torch.relu(self.conv1(x)))

         x = self.pool2(torch.relu(self.conv2(x)))

         x = x.view(x.size(0), -1)  # flatten the output of the pooling layers

         x = torch.relu(self.fc1(x))

         x = torch.relu(self.fc2(x))

         x = self.fc3(x)

         return x





def CNN_model(image, model, device):

    image = cv2.resize(image, (760,560))

    TARGET_WIDTH = 32

    TARGET_HEIGHT = 32

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)




    # Tính trung bình độ tương phản của các điểm trên ảnh

    contrast_mean = np.mean(gray)




    # Thực hiện quyết định dựa trên giá trị trung bình độ tương phản

    threshold_low = 130  # Ngưỡng thấp để sử dụng biện pháp tăng độ tương phản

    threshold_high = 70  # Ngưỡng cao để sử dụng biện pháp tăng độ tương phản




    if threshold_high < contrast_mean < threshold_low:

        # Sử dụng biện pháp tăng độ tương phản

        enhanced_gray = cv2.equalizeHist(gray)




        # Thay đổi ngưỡng tăng độ tương phản

        threshold_bright = 0.43  # Ngưỡng cho điểm ảnh trắng




        # Tăng độ trắng cho các điểm ảnh trắng có giá trị < threshold_bright * 255

        enhanced_gray[enhanced_gray < threshold_bright * 255] = 0

    else:

        enhanced_gray = gray




    # Làm mờ ảnh để làm giảm nhiễu

    blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)




    # Ngưỡng hóa ảnh để chuyển đổi thành ảnh nhị phân

    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)




    # Kết nối thành phần liên thông

    _, labels = cv2.connectedComponents(thresholded)

    mask = np.zeros(thresholded.shape, dtype="uint8")

    total_pixels = image.shape[0] * image.shape[1]

    lower = total_pixels // 100  # heuristic param, can be fine-tuned if necessary

    upper = total_pixels // 30




    # Xác định các vùng quan tâm

    for (i, label) in enumerate(np.unique(labels)):

        # Bỏ qua nhãn nền

        if label == 0:

            continue




        # Tạo mask cho nhãn hiện tại

        labelMask = np.zeros(thresholded.shape, dtype="uint8")

        labelMask[labels == label] = 255




        # Đếm số điểm ảnh trong vùng quan tâm

        numPixels = cv2.countNonZero(labelMask)




        # Nếu số điểm ảnh nằm trong một khoảng xác định, thì thêm vào mask chung

        if numPixels > lower and numPixels < upper:

            mask = cv2.add(mask, labelMask)




    # Tìm các contour trên mask

    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]




    def compare(rect1, rect2):

        if abs(rect1[1] - rect2[1]) > 27:

            return rect1[1] - rect2[1]

        else:

            return rect1[0] - rect2[0]

    boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare))

    result = ''

    for rect in boundingBoxes:

        # Get the coordinates from the bounding box

        x,y,w,h = rect

        # Crop the character from the mask

        # and apply bitwise_not because in our training data for pre-trained model

        # the characters are black on a white background

        crop = mask[y:y+h, x:x+w]

        #pic = cv2.countNonZero(crop)




        h= crop.shape[0]

        w= crop.shape[1]





        if 4> h/w >= 2.5 :




            crop = cv2.bitwise_not(crop)




            rows = crop.shape[0]




            columns = crop.shape[1]




            paddingY = int(0 * rows)




            paddingX = int(0.15 * columns)






        # Apply padding to make the image fit for neural network model




            crop = cv2.copyMakeBorder(crop, paddingY, paddingY, paddingX, paddingX, cv2.BORDER_CONSTANT, None, 255)




            crop = cv2.bitwise_not(crop)





        # Get the number of rows and columns for each cropped image

        # and calculate the padding to match the image input of pre-trained model

        rows = crop.shape[0]

        columns = crop.shape[1]

        # Apply padding to make the image fit for neural network model

        #crop = cv2.copyMakeBorder(crop, paddingY, paddingY, paddingX, paddingX, cv2.BORDER_CONSTANT, None, 255)

        # Convert and resize image

        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)




        crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))





        # crop = crop.astype("float") / 255.0

        transform = transforms.ToTensor()

        crop = transform(crop)

        #crop = np.expand_dims(crop, axis=0)

        data = torch.unsqueeze(crop, dim=0) # unsqueeze data

        data = data.to(device)






        output = model(data)

        output = F.log_softmax(output, dim=1) # log softmax, chú ý dim

        pred = output.argmax(dim=1, keepdim=True) # argmax, chú ý keepdim, dim=1

        label= pred[0][0].detach().cpu().numpy()

        final = classes[label]

        result += final

    return result
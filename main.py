import os
import time

import cv2
from cv2 import resize
import numpy as np

from ClsTemplateModel import TemplateModel

CAMERA_ID = 0
DELAY = 5
THRESH_MACH = 0.9

def addTemplateImages(image:np.ndarray, tempModel:TemplateModel):
    image = cv2.resize(image, (500, 500))
    tempModel.addTempImage(image)
    height = image.shape[0]
    width = image.shape[1]
    center = (int(width/2), int(height/2))
    for i in range(10, 40, 10):
        angle = i
        trans = cv2.getRotationMatrix2D(center, angle , 1.0)
        new = cv2.warpAffine(image, trans, (width,height), borderValue=(255, 255, 255))
        tempModel.addTempImage(new)
        angle = -i
        trans = cv2.getRotationMatrix2D(center, angle , 1.0)
        new = cv2.warpAffine(image, trans, (width,height), borderValue=(255, 255, 255))
        tempModel.addTempImage(new)
    # new = cv2.rotate(image, cv2.ROTATE_180)
    # tempModel.addTempImage(new)
    return tempModel


def main():
    # 判定に使う領域
    roi_size = (400, 400)
    roi_lt = (440, 250)
    roi_rb = (roi_lt[0]+roi_size[0], roi_lt[1]+roi_size[1])

    # テンプレートの登録
    temps = []
    img1 = cv2.imread('temp_images/image2-1.jpg')
    img1 = resize(img1, roi_size)
    temps.append(TemplateModel(1, img1))
    img2 = cv2.imread('temp_images/image2-2.jpg')
    img2 = resize(img2, roi_size)
    temps.append(TemplateModel(2, img2))
    img3 = cv2.imread('temp_images/image2-3.jpg')
    img3 = resize(img3, roi_size)
    temps.append(TemplateModel(3, img3))
    img4 = cv2.imread('temp_images/image2-4.jpg')
    img4 = resize(img4, roi_size)
    temps.append(TemplateModel(4, img4))
    

    # カメラ設定
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    ret, frame_rgb = cap.read()


    isMachImage = False
    while cap.isOpened():
        if ret:
            frame_gray = frame_rgb[roi_lt[1]:roi_rb[1], roi_lt[0]:roi_rb[0]]
            frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)

            max_val = -1
            max_id = 0
            for temp in temps:
                val = temp.matches(frame_gray)
                if val > max_val:
                    max_val = val
                    max_id = temp.getId()

            threshold = THRESH_MACH
            if max_val >= threshold:
                isMachImage = True

            cv2.rectangle(frame_rgb, roi_lt, roi_rb, (255, 0, 0), 3)
            frame_rgb = cv2.flip(frame_rgb, 1)
            cv2.imshow('camera', frame_rgb)
            print(f'id:{max_id}, val:{max_val}')
        
        if isMachImage or (cv2.waitKey(DELAY) & 0xFF == ord('q')):
            break

        ret, frame_rgb = cap.read()

    cap.release()
    time.sleep(3)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
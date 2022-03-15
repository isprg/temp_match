import os
import time

import cv2
import numpy as np

from ClsTemplateModel import TemplateModel

CAMERA_ID = 0
DELAY = 5
THRESH_MACH = 1.0

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
    roi_lt = (440, 200)
    roi_rb = (roi_lt[0]+roi_size[0], roi_lt[1]+roi_size[1])

    # テンプレートの登録
    img1 = cv2.imread('temp_images/image2-1.jpg')
    temp1 = TemplateModel(1)
    temp1 = addTemplateImages(img1, temp1)
    # temp1.showTempImage(5)

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

            val, loc, w, h= temp1.matches(frame_gray)
            # val = 0.9

            threshold = THRESH_MACH
            if val >= threshold:
                top_left = loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                # cv2.rectangle(frame_rgb, top_left, bottom_right, color=(0,0,255), thickness=3)
                print(f'({top_left}, {bottom_right})')
                isMachImage = True

            cv2.rectangle(frame_rgb, roi_lt, roi_rb, (255, 0, 0), 3)
            frame_rgb = cv2.flip(frame_rgb, 1)
            cv2.imshow('camera', frame_rgb)
            print(f'id:{temp1.getId()}, val:{val}')
        
        if isMachImage or (cv2.waitKey(DELAY) & 0xFF == ord('q')):
            break

        ret, frame_rgb = cap.read()

    cap.release()
    time.sleep(1)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
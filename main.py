import os
import time

import cv2
import numpy as np

import poc

CAMERA_ID = 0
DELAY = 5
THRESH_MACH = 0.9

def inputTemplates(dirName):
    """テンプレート画像を全て読み込む関数 

    Parameters:
        dirName(str): テンプレート画像が入ったﾃﾞｨﾚｸﾄﾘ名
    Returns:
        images(list): 全てのテンプレート画像のnumpy形式データが入ったリスト
    """
    images = []
    for fileName in os.listdir(dirName):
        image = cv2.imread(os.path.join(dirName, fileName))
        images.append(image)
    return images

def main():
    temps_rgb = inputTemplates('temp_images')
    temps_gray = []
    for img in temps_rgb:
        temps_gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    ret, frame_rgb = cap.read()

    isMachImage = False
    while cap.isOpened():
        if ret:
            frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)

            val_mach = -1
            loc_mach = (0, 0)
            w_mach = 0
            h_mach = 0
            for temp in temps_gray:
                val, loc_x, loc_y = poc.poc(temp, frame_gray)
                if val > val_mach:
                    val_mach = val
                    loc_mach = (loc_x, loc_y)
                    w_mach, h_mach = temp.shape[::-1]

            threshold = THRESH_MACH
            if val_mach >= threshold:
                top_left = loc_mach
                bottom_right = (top_left[0] + w_mach, top_left[1] + h_mach)
                cv2.rectangle(frame_rgb, top_left, bottom_right, color=(0,0,255), thickness=3)
                print(f'{top_left}, {bottom_right}')
                isMachImage = True

            frame_rgb = cv2.flip(frame_rgb, 1)
            cv2.imshow('camera', frame_rgb)
            print(f'isMach:{isMachImage}, val:{val_mach}')
        
        if isMachImage or (cv2.waitKey(DELAY) & 0xFF == ord('q')):
            break

        ret, frame_rgb = cap.read()

    cap.release()
    time.sleep(3)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
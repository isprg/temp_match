import os
import time

import cv2
import numpy as np

CAMERA_ID = 0
DELAY = 5
THRESH_MACH = 0.85

def inputTemplates(dirName):
    """テンプレート画像を全て読み込む関数 

    Args:
        dirName(str): テンプレート画像が入ったﾃﾞｨﾚｸﾄﾘ名
    Return:
        images(list): 全てのテンプレート画像のnumpy形式データが入ったリスト
    """
    images = []
    for fileName in os.listdir(dirName):
        image = cv2.imread(os.path.join(dirName, fileName))
        images.append(image)
    return images

def templateMatch(img, temp):
    """テンプレートマッチングを行い類似度と場所を返す

    Args:
        img: テンプレートを探すグレースケール画像
        temp: テンプレートのグレースケール画像
    Return:
        val: 最大の類似度の値
        loc: 類似度が最大の左上の座標
    """
    res = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_val, max_loc


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
            loc_mach = [0, 0]
            w_mach = 0
            h_mach = 0
            for temp in temps_gray:
                val, loc = templateMatch(frame_gray, temp)
                if val > val_mach:
                    val_mach = val
                    loc_mach = loc
                    w_mach, h_mach = temp.shape[::-1]

            threshold = THRESH_MACH
            if val_mach >= threshold:
                top_left = loc_mach
                bottom_right = (top_left[0] + w_mach, top_left[1] + h_mach)
                frame_rgb = cv2.rectangle(frame_rgb, top_left, bottom_right, (255, 0, 0), 3)
                print(f'{top_left}, {bottom_right}')
                isMachImage = True

            frame_rgb = cv2.flip(frame_rgb, 1)
            cv2.imshow("camera", frame_rgb)
            print(f'isMach:{isMachImage}, val:{val_mach}')
        
        if isMachImage or (cv2.waitKey(DELAY) & 0xFF == ord('q')):
            break

        ret, frame_rgb = cap.read()

    time.sleep(5)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
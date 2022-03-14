import cv2
import numpy as np

def matchFunc(self, img:np.ndarray, temp:np.ndarray):
        """テンプレートマッチングを行い類似度と場所を返す

        Parameters:
            img: テンプレートを探すグレースケール画像
            temp: テンプレートのグレースケール画像
        Returns:
            val: 最大の類似度の値
            loc: 類似度が最大の左上の座標
        """
        res = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        return max_val, max_loc
from socket import fromshare
import cv2
import numpy as np

from TemplateMatch import matchFunc

class TemplateModel:
    def __init__(self, id:int, image:np.ndarray=None):
        self.id = id
        self.temps = []
        self.val_mach = -1
        self.loc_mach = (0, 0)
        self.w_mach = 0
        self.h_mach = 0
        if image is not None:
            self.addTempImage(image)

    def getId(self):
        return self.id

    def showTempImage(self, index:int=0, windowname:str='template'):
        try:
            cv2.imshow(windowname, self.temps[index])
        except:
            print('対応インデックスに画像が登録されていません')

    def addTempImage(self, image:np.ndarray):
        self.temps.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        return self.temps[-1]

    def Templatematches(self, img:np.ndarray):
        """複数のテンプレートを用い、最も高い類似度等を割り出す

        Parameters:
            img: テンプレートを探すグレースケール画像
        Returns:
            val: 複数テンプレートとの最も高い類似度の値
            loc: 最も高い類似度を出した左上の座標（タプル型）
            width: 最も高い類似度を算出したテンプレートの横幅
            height: 最も高い類似度を算出したテンプレートの縦幅
        """
        for temp in self.temps:
            val, loc = matchFunc(img, temp)
            if val > self.val_mach:
                self.val_mach = val
                self.loc_mach = (loc[0], loc[1])
                self.w_mach, self.h_mach = temp.shape[::-1]
        return self.val_mach, self.loc_mach, self.w_mach, self.h_mach

    def drawRectangle(self, image:np.ndarray):
        left_top = self.loc_mach
        right_bottom = (self.left_top[0] + self.w_mach, self.left_top[1] + self.h_mach)
        image = cv2.rectangle(image, left_top, right_bottom, (255, 0, 0), 3)
        return image

import cv2
import numpy as np

from TemplateMatch import matchFunc

class TemplateModel:
    def __init__(self, id:int, image:np.ndarray=None):
        self.id = id
        self.temps = []
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

    def matches(self, img:np.ndarray):
        """複数のテンプレートを用い、最も高い類似度等を割り出す

        Parameters:
            img: テンプレートを探すグレースケール画像
        Returns:
            val: 複数テンプレートとの最も高い類似度の値
        """
        val_match = -1
        for temp in self.temps:
            val, loc = matchFunc(img, temp)
            if val > val_match:
                val_match = val
            elif val < val_match:
                break
        return val_match

    # def drawRectangle(self, image:np.ndarray):
    #     left_top = self.loc_match
    #     right_bottom = (self.left_top[0] + self.w_match, self.left_top[1] + self.h_match)
    #     image = cv2.rectangle(image, left_top, right_bottom, (255, 0, 0), 3)
    #     return image

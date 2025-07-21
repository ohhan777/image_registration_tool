import cv2
import numpy as np
from pathlib import Path

def imread_unicode(path):
    # np.fromfile로 바이너리 읽고, cv2.imdecode로 디코딩
    arr = np.fromfile(str(path), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def imwrite_unicode(path, img):
    # 확장자 추출
    ext = str(path)[str(path).rfind('.'):]
    # cv2.imencode로 인코딩 후, np.tofile로 저장
    result, encoded_img = cv2.imencode(ext, img)
    if result:
        encoded_img.tofile(str(path))
        return True
    return False




filename = Path('LCM00001_PS3_K3_Siheung_20151009_Google.png')
img = imread_unicode(filename)


# crop the image by 1024x1024 from the center

# get the center of the image
center_x = img.shape[1] // 2
center_y = img.shape[0] // 2

# crop the image
cropped_img = img[center_y-512:center_y+512, center_x-512:center_x+512]

# save the cropped image
imwrite_unicode(filename.with_name(filename.stem + '_1024.png'), cropped_img)

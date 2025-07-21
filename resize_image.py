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

def process_image_to_1024(img):
    """
    이미지를 1024x1024로 처리하는 함수
    - 이미지가 1024x1024보다 크면 중앙에서 1024x1024로 자르기
    - 이미지가 1024x1024보다 작으면 중앙을 기준으로 정사각형으로 1024x1024로 확대
    """
    target_size = 1024
    height, width = img.shape[:2]
    
    # 이미지가 1024x1024보다 큰 경우: 중앙에서 1024x1024로 자르기
    if height >= target_size and width >= target_size:
        center_x = width // 2
        center_y = height // 2
        cropped_img = img[center_y-target_size//2:center_y+target_size//2, 
                         center_x-target_size//2:center_x+target_size//2]
        return cropped_img
    
    # 이미지가 1024x1024보다 작은 경우: 중앙을 기준으로 정사각형으로 1024x1024로 확대
    else:
        # 정사각형으로 만들기 위해 더 작은 차원을 기준으로 정사각형 생성
        min_dim = min(height, width)
        
        # 중앙에서 정사각형 영역 추출
        center_x = width // 2
        center_y = height // 2
        half_size = min_dim // 2
        
        # 정사각형 영역 추출
        square_img = img[center_y-half_size:center_y+half_size, 
                        center_x-half_size:center_x+half_size]
        
        # 1024x1024로 리사이즈
        resized_img = cv2.resize(square_img, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        return resized_img

filename = Path('LCM00001_PS3_K3_Siheung_20151009_Google.png')
img = imread_unicode(filename)

# 이미지를 1024x1024로 처리
processed_img = process_image_to_1024(img)

# 처리된 이미지 저장
imwrite_unicode(filename.with_name(filename.stem + '_1024.png'), processed_img)

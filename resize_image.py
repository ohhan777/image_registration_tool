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
    - 이미지가 1024x1024보다 크면 중앙에서 1:1 aspect ratio로 자르고 1024x1024로 다운스케일
    - 이미지가 1024x1024보다 작으면 중앙을 기준으로 1:1 aspect ratio로 만들고 1024x1024로 업스케일
    """
    target_size = 1024
    height, width = img.shape[:2]
    
    # 중앙 좌표 계산
    center_x = width // 2
    center_y = height // 2
    
    # 1:1 aspect ratio를 위해 더 큰 차원을 기준으로 정사각형 생성
    # 이렇게 하면 고해상도 이미지에서 최대한의 정보를 보존
    square_size = max(height, width)
    
    # 정사각형 영역의 시작과 끝 좌표 계산
    half_size = square_size // 2
    
    # 이미지 경계를 벗어나지 않도록 조정
    start_y = max(0, center_y - half_size)
    end_y = min(height, center_y + half_size)
    start_x = max(0, center_x - half_size)
    end_x = min(width, center_x + half_size)
    
    # 실제 추출할 영역의 크기 계산
    actual_height = end_y - start_y
    actual_width = end_x - start_x
    
    # 정사각형이 되도록 패딩 추가 (필요한 경우)
    if actual_height != actual_width:
        # 더 작은 차원을 더 큰 차원에 맞춤
        max_dim = max(actual_height, actual_width)
        
        # 새로운 정사각형 이미지 생성 (검은색 배경)
        square_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        
        # 원본 이미지를 중앙에 배치
        y_offset = (max_dim - actual_height) // 2
        x_offset = (max_dim - actual_width) // 2
        
        square_img[y_offset:y_offset+actual_height, x_offset:x_offset+actual_width] = \
            img[start_y:end_y, start_x:end_x]
    else:
        # 이미 정사각형인 경우
        square_img = img[start_y:end_y, start_x:end_x]
    
    # 1024x1024로 리사이즈
    resized_img = cv2.resize(square_img, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    return resized_img

filename = Path('LCM00001_PS3_K3_Siheung_20151009_Google.png')
img = imread_unicode(filename)

# 이미지를 1024x1024로 처리
processed_img = process_image_to_1024(img)

# 처리된 이미지 저장
imwrite_unicode(filename.with_name(filename.stem + '_1024.png'), processed_img)

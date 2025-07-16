import os
import cv2
import numpy as np
from pathlib import Path

def read_corresponding_points(file_path1, file_path2):
    """Parse corresponding points from two files"""
    points1, points2 = {}, {}
    
    def read_file(file_path):
        points = {}
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().replace(',', '').split()
                if len(parts) >= 3:
                    try:
                        key, x, y = parts[0], float(parts[1]), float(parts[2])
                        points[int(key)] = [x, y]
                    except ValueError:
                        print(f"Skipping invalid line: {line.strip()}")
        return points

    points1 = read_file(file_path1)
    points2 = read_file(file_path2)

    # Find corresponding points
    common_keys = sorted(set(points1.keys()) & set(points2.keys()))  # sorted for consistent ordering
    corresponding_points1 = [points1[k] for k in common_keys]
    corresponding_points2 = [points2[k] for k in common_keys]

    return np.array(corresponding_points1), np.array(corresponding_points2), common_keys

def register_images(img1_path, img2_path, points1, points2):
    """
    Register image2 to image1 using affine transform
    
    Args:
        img1_path: Path to reference image
        img2_path: Path to image that needs to be transformed
        points1: Corresponding points from image1
        points2: Corresponding points from image2
    
    Returns:
        transformed_img: Registered image
        transform_matrix: 2x3 affine transformation matrix
    """
    # Read images
    img1 = imread_unicode(img1_path)
    img2 = imread_unicode(img2_path)
    
    if img1 is None or img2 is None:
        raise ValueError("Failed to load images")
    
    # Convert points to float32
    src_points = points2.astype(np.float32)
    dst_points = points1.astype(np.float32)
    
    # Calculate affine transform matrix
    transform_matrix = cv2.estimateAffinePartial2D(src_points, dst_points)[0]
    
    # Apply affine transform to image2
    transformed_img = cv2.warpAffine(
        img2,
        transform_matrix,
        (img1.shape[1], img1.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # 포인트 변환: cv2.transform 사용 (2x3 행렬 직접 지원)
    points2_reshaped = np.array(points2, dtype=np.float32).reshape(-1, 1, 2)
    registered_points2 = cv2.transform(points2_reshaped, transform_matrix)
    registered_points2 = registered_points2.reshape(-1, 2)
    
    return transformed_img, transform_matrix, registered_points2

def draw_point_matches(overlay_img, points1, points2, keys):
    """
    overlay_img: numpy array (이미지)
    points1: (N, 2) 원본 좌표
    points2: (N, 2) 변환(registered)된 좌표
    keys: list of keys for the points
    """
    import numpy as np
    import cv2
    for key, ((x1, y1), (x2, y2)) in zip(keys, zip(points1, points2)):
        dist = np.hypot(x1 - x2, y1 - y2)
        color = (0, 255, 0) if dist <= 2.0 else (0, 0, 255)  # 2픽셀 이내: 녹색, 초과: 빨간색
        cv2.circle(overlay_img, (int(round(x2)), int(round(y2))), 3, color, -1)
        # 숫자(순서) 표시 (흰색, 점 오른쪽 위)
        cv2.putText(
            overlay_img,
            str(key),
            (int(round(x2)) + 4, int(round(y2)) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),  # 흰색 (BGR)
            1,
            cv2.LINE_AA
        )

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

# Parse corresponding points
reg_file1 = Path('C:/Users/tsg03/Desktop/png_points/LCM00002_PS3_K3_CHIAYI_20191213.txt')
reg_file2 = Path('C:/Users/tsg03/Desktop/png_points/LCM00002_PS3_K3_CHIAYI_20220222.txt')

try:
    points1, points2, common_keys = read_corresponding_points(reg_file1, reg_file2)

    if len(points1) == 0 or len(points2) == 0:
        raise ValueError("No valid points found in one or both files")

    # 실제 사용 예시
    img1_path = Path("C:/Users/tsg03/Desktop/original_image") / reg_file1.name.replace(".txt", ".png")
    img2_path = Path("C:/Users/tsg03/Desktop/original_image") / reg_file2.name.replace(".txt", ".png")
    
    registered_img, transform_matrix, registered_points2 = register_images(
        img1_path,
        img2_path,
        points1,
        points2
    )
    
    # 결과 저장
    cv2.imwrite("registered_image.png", registered_img)
    print("Affine Transform Matrix:")
    print(transform_matrix)
    
    # Optional: 결과 시각화
    ref_img = imread_unicode(img1_path)
    ref_img_copy = ref_img.copy()
    registered_img_copy = registered_img.copy()

    # points on the reference image with numbers
    for key, point in zip(common_keys, points1):
        x, y = int(point[0]), int(point[1])
        cv2.circle(ref_img_copy, (x, y), 3, (0, 255, 255), -1)
        cv2.putText(
            ref_img_copy,
            str(key),
            (x + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),  # 흰색
            1,
            cv2.LINE_AA
        )
    
    # points on the registered image with numbers
    for key, point in zip(common_keys, registered_points2):
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(registered_img_copy, (x, y), 3, (0, 255, 255), -1)
        cv2.putText(
            registered_img_copy,
            str(key),
            (x + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),  # 흰색
            1,
            cv2.LINE_AA
        )

    imwrite_unicode("reference_image.png", ref_img_copy)
    imwrite_unicode("registered_image.png", registered_img_copy)

    # Overlay image
    overlay_img = cv2.addWeighted(ref_img, 0.5, registered_img, 0.5, 0)
    
    draw_point_matches(overlay_img, points1, registered_points2, common_keys)

    imwrite_unicode("overlay_image.png", overlay_img)
    
except Exception as e:
    print(f"Error occurred: {str(e)}")

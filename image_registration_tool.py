import os
# GeoTIFF(google_ref .tif)의 지리 메타데이터 태그를 Qt TIFF 플러그인이
# 인식하지 못해 찍는 "Unknown field with tag ..." 경고를 끈다.
# (이미지 로드에는 문제가 없으며 콘솔 경고만 억제)
os.environ.setdefault("QT_LOGGING_RULES", "qt.imageformats.tiff=false")

import sys
import typing
import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsLineItem, QGraphicsTextItem,
    QToolBar, QFileDialog, QInputDialog, QLineEdit, QMessageBox
)
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QIcon, QAction, QKeySequence, QPainterPath
from PIL import Image, ImageDraw, ImageFont
import re

import cv2
from PyQt6.QtGui import QImage

class ImageViewer(QGraphicsView):
    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')

    def __init__(self):
        super().__init__()
        self.init_ui()

        # 이미지의 중심을 중심으로 확대/축소하도록 설정
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        # 마우스 휠 이벤트 핸들링
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 마우스 추적 활성화
        self.setMouseTracking(True)

        # 드래그 앤 드롭 활성화
        self.setAcceptDrops(True)

    def init_ui(self):
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.image_item = None
        self.cross_items = []
        self.number_items = []
        self.coordinates = []
        self.number_count = 0
        self._dirty = False

        self.zoom_factor = 1.0
        self.zoom_step = 0.1
        self.min_zoom = 0.05
        self.max_zoom = 2.0
        self.setMinimumSize(400, 400)
        
        # 드래그 스크롤 관련 변수
        self.last_pan_point = None
        self.is_panning = False

        # 좌클릭 드래그/클릭 구분용 변수
        self._left_press_pos = None
        self._left_moved = False
        self._click_drag_threshold = 4  # 이 픽셀 이상 움직이면 클릭이 아닌 드래그로 간주
        
        # Undo 관련 변수
        self.undo_stack = []

    def _mark_dirty(self, dirty=True):
        self._dirty = dirty
        window = self.window()
        if isinstance(window, Image_Window):
            window.update_title()

    def load_image(self, file_name):
        self.scene.clear()
        self.cross_items.clear()
        self.number_items.clear()
        self.coordinates.clear()
        self.number_count = 0
        self._mark_dirty(False)
        pixmap = QPixmap(file_name)
        if not pixmap.isNull():
            self.image_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.image_item)

            self.setSceneRect(pixmap.rect().x(), pixmap.rect().y(), pixmap.rect().width(), pixmap.rect().height())
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        try:
            update_registration_button()
        except NameError:
            pass

    def load_from_numpy(self, np_img):
        img_rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        height, width, _ = img_rgb.shape
        bytes_per_line = 3 * width
        qimage = QImage(img_rgb.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.scene.clear()
        self.cross_items.clear()
        self.number_items.clear()
        self.coordinates.clear()
        self.number_count = 0
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._mark_dirty(False)

    # 좌표 저장
    def save_coordinates_image(self, file_name):
        if not self.coordinates:
            return

        image = Image.new("RGB", (int(self.scene.width()), int(self.scene.height())), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        scene_pixmap = QPixmap(int(self.scene.width()), int(self.scene.height()))
        scene_pixmap.fill(Qt.GlobalColor.white)
        painter = QPainter(scene_pixmap)
        self.scene.render(painter)
        painter.end()
        # PIL.Image.fromqpixmap은 PyQt6에서 지원하지 않으므로, QPixmap을 bytes로 변환 후 PIL로 변환 필요
        image_bytes = scene_pixmap.toImage().bits().asstring(scene_pixmap.toImage().byteCount())
        pil_image = Image.frombytes("RGBA", (scene_pixmap.width(), scene_pixmap.height()), image_bytes)
        image.paste(pil_image, (0, 0))

        font = ImageFont.load_default()

        for i, (x, y) in enumerate(self.coordinates, start=1):
            draw.text((int(x) + 5, int(y) - 5), f"{i}", fill=(255, 0, 0), font=font)

        image.save(file_name)
        try:
            update_registration_button()
        except NameError:
            pass

    def plus_image(self):
        self.zoom_factor += self.zoom_step
        self.scale(self.zoom_factor, self.zoom_factor)
        self._notify_sync()

    def minus_image(self):
        if self.zoom_factor > self.zoom_step:
            self.zoom_factor -= self.zoom_step
            self.scale(self.zoom_factor, self.zoom_factor)
            self._notify_sync()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Ctrl 키가 눌려있으면 패닝 모드로 전환
            if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                self.last_pan_point = event.position().toPoint()
                self.is_panning = True
                return

            # 플레인 좌클릭: 드래그와 클릭을 구분하기 위해 눌린 위치만 기록하고
            # 실제 처리(점 찍기/레이블 편집)는 마우스 릴리스 시점으로 미룬다.
            self._left_press_pos = event.position().toPoint()
            self._left_moved = False
        elif event.button() == Qt.MouseButton.RightButton:
            pos = self.mapToScene(event.position().toPoint())
            # 클릭한 좌표 주변에 있는 좌표를 삭제
            if self.remove_coordinates(pos):
                return
        elif event.button() == Qt.MouseButton.MiddleButton:
            # 중간 마우스 버튼으로 패닝 시작
            self.last_pan_point = event.position().toPoint()
            self.is_panning = True

    def _handle_left_click(self, event):
        """드래그가 아닌 실제 클릭일 때: 기존 레이블 편집 또는 새 점 찍기."""
        pos = self.mapToScene(event.position().toPoint())

        # 사용자가 이미 존재하는 숫자 레이블을 클릭했는지 확인합니다.
        for i, (x, y) in enumerate(self.coordinates):
            distance = (x - pos.x())**2 + (y - pos.y())**2
            if distance < 9:
                new_label, ok = QInputDialog.getText(self, '레이블 수정', '좌표에 대한 새로운 레이블을 입력하세요:', QLineEdit.EchoMode.Normal, self.number_items[i].toPlainText())
                if ok:
                    self.modify_coordinate_label(i, new_label)
                return
        self.Click_Coordinate(pos)

    def Click_Coordinate(self, pos):
        # Undo를 위한 현재 상태 저장
        self.save_state_for_undo()
        
        cross_size = 7
        pen = QPen(QColor(255, 0, 0))
        cross_item1 = QGraphicsLineItem(pos.x() - cross_size / 2, pos.y(), pos.x() + cross_size / 2, pos.y())
        cross_item2 = QGraphicsLineItem(pos.x(), pos.y() - cross_size / 2, pos.x(), pos.y() + cross_size / 2)
        cross_item1.setPen(pen)
        cross_item2.setPen(pen)

        self.cross_items.append(cross_item1)
        self.cross_items.append(cross_item2)

        self.scene.addItem(cross_item1)
        self.scene.addItem(cross_item2)

        number_item = QGraphicsTextItem(str(self.number_count + 1))
        number_item.setPos(pos.x() + cross_size , pos.y() - cross_size )
        number_item.setDefaultTextColor(QColor(Qt.GlobalColor.red))
        font = QFont()
        font.setPointSize(7)
        number_item.setFont(font)
        self.scene.addItem(number_item)
        self.number_items.append(number_item)
        self.number_count += 1
        self.coordinates.append(((pos.x()), (pos.y())))
        self._mark_dirty(True)
        try:
            update_registration_button()
        except NameError:
            pass

    # 좌표 전체 삭제
    def remove_cross_items(self):
        for item in self.cross_items:
            self.scene.removeItem(item)
        for n_item in self.number_items:
            self.scene.removeItem(n_item)

        self.coordinates = []
        self.number_items = []
        self.cross_items = []
        self.number_count = 0
        self._mark_dirty(True)
        try:
            update_registration_button()
        except NameError:
            pass

    def remove_cross_one_item(self, index):
        if 0 <= index < len(self.cross_items) and 0 <= index < len(self.number_items):
            cross_item1 = self.cross_items.pop(index * 2)
            cross_item2 = self.cross_items.pop(index * 2)
            self.scene.removeItem(cross_item1)
            self.scene.removeItem(cross_item2)
            self.scene.removeItem(self.number_items[index])
            self.number_items.pop(index)
            self.number_count -= 1
        try:
            update_registration_button()
        except NameError:
            pass

    # 좌표 개별 삭제
    def remove_coordinates(self, pos):
        for i, (x, y) in enumerate(self.coordinates):
            distance = (x - pos.x())**2 + (y - pos.y())**2
            if distance < 9:
                # Undo를 위한 현재 상태 저장
                self.save_state_for_undo()
                self.remove_cross_one_item(i)
                self.coordinates.pop(i)
                self._mark_dirty(True)
                try:
                    update_registration_button()
                except NameError:
                    pass
                return True
        return False
    
    # Undo 기능을 위한 상태 저장
    def save_state_for_undo(self):
        state = {
            'coordinates': self.coordinates.copy(),
            'number_count': self.number_count,
            'cross_items_data': [],
            'number_items_data': []
        }
        
        # 크로스 아이템들의 데이터 저장
        for i in range(0, len(self.cross_items), 2):
            if i + 1 < len(self.cross_items):
                cross1 = self.cross_items[i]
                cross2 = self.cross_items[i + 1]
                state['cross_items_data'].append({
                    'cross1_line': cross1.line(),
                    'cross2_line': cross2.line()
                })
        
        # 숫자 아이템들의 데이터 저장
        for item in self.number_items:
            state['number_items_data'].append({
                'text': item.toPlainText(),
                'pos': item.pos()
            })
        
        self.undo_stack.append(state)
        # Undo 스택 크기 제한 (메모리 절약)
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
    
    # Undo 기능 실행
    def undo(self):
        if not self.undo_stack:
            return
        
        # 현재 모든 아이템 제거
        for item in self.cross_items:
            self.scene.removeItem(item)
        for item in self.number_items:
            self.scene.removeItem(item)
        
        # 이전 상태 복원
        state = self.undo_stack.pop()
        self.coordinates = state['coordinates']
        self.number_count = state['number_count']
        self.cross_items = []
        self.number_items = []
        
        # 크로스 아이템들 복원
        pen = QPen(QColor(255, 0, 0))
        for cross_data in state['cross_items_data']:
            cross_item1 = QGraphicsLineItem(cross_data['cross1_line'])
            cross_item2 = QGraphicsLineItem(cross_data['cross2_line'])
            cross_item1.setPen(pen)
            cross_item2.setPen(pen)
            self.cross_items.append(cross_item1)
            self.cross_items.append(cross_item2)
            self.scene.addItem(cross_item1)
            self.scene.addItem(cross_item2)
        
        # 숫자 아이템들 복원
        for number_data in state['number_items_data']:
            number_item = QGraphicsTextItem(number_data['text'])
            number_item.setPos(number_data['pos'])
            number_item.setDefaultTextColor(QColor(Qt.GlobalColor.red))
            font = QFont()
            font.setPointSize(7)
            number_item.setFont(font)
            self.number_items.append(number_item)
            self.scene.addItem(number_item)
        self._mark_dirty(True)
        try:
            update_registration_button()
        except NameError:
            pass

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(self.IMAGE_EXTENSIONS):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path.lower().endswith(self.IMAGE_EXTENSIONS):
                    window = self.window()
                    if isinstance(window, Image_Window):
                        window.open_image_auto(file_path)
                    event.acceptProposedAction()
                    return
        event.ignore()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Z and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.undo()
        else:
            super().keyPressEvent(event)

    def mouseMoveEvent(self, event):
        # 플레인 좌클릭 후 임계값 이상 움직이면 클릭이 아닌 드래그(패닝)로 전환
        if (self._left_press_pos is not None and not self._left_moved
                and event.buttons() & Qt.MouseButton.LeftButton):
            moved = (event.position().toPoint() - self._left_press_pos).manhattanLength()
            if moved > self._click_drag_threshold:
                self._left_moved = True
                self.is_panning = True
                self.last_pan_point = event.position().toPoint()

        if self.is_panning and self.last_pan_point is not None:
            # 드래그 거리 계산
            delta = event.position().toPoint() - self.last_pan_point
            self.last_pan_point = event.position().toPoint()

            # 스크롤바 이동
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._notify_sync()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = False
            self.last_pan_point = None
        elif event.button() == Qt.MouseButton.LeftButton:
            if self.is_panning:
                # 드래그였으므로 점을 찍지 않고 패닝만 종료
                self.is_panning = False
                self.last_pan_point = None
            elif self._left_press_pos is not None and not self._left_moved:
                # 움직임이 거의 없었던 실제 클릭 → 점 찍기/레이블 편집 수행
                self._handle_left_click(event)
            self._left_press_pos = None
            self._left_moved = False

        super().mouseReleaseEvent(event)

    def _notify_sync(self):
        """뷰 변경 후 파트너 윈도우에 동기화 알림"""
        window = self.window()
        if isinstance(window, Image_Window) and not window._syncing:
            window._sync_to_partner()

    def wheelEvent(self, event):
        # 마우스 휠 이벤트를 감지하여 이미지 확대/축소
        self.zoom_level = 1.0
        zoom_out_scale = 0.9
        zoom_in_scale = 1.1

        if event.angleDelta().y() > 0:
            # 양수인 경우, 이미지 확대
            self.zoom_level *= zoom_in_scale
        else:
            # 음수인 경우, 이미지 축소
            self.zoom_level *= zoom_out_scale

        # 이미지의 최소 및 최대 확대/축소 비율을 설정합니다. 필요에 따라 조정할 수 있습니다.
        min_zoom = 0.1
        max_zoom = 10.0
        self.zoom_level = max(min_zoom, min(max_zoom, self.zoom_level))

        # 이미지를 확대/축소합니다.
        self.scale(self.zoom_level, self.zoom_level)
        self._notify_sync()

    # txt 파일로 좌표 데이터 저장
    def save_coordinates_to_txt(self, file_name):
        if not self.coordinates:
            return 
        
        with open(file_name, 'w') as file:
            for number_item, (x, y) in zip(self.number_items, self.coordinates):
                label = number_item.toPlainText()
                if (x, y) != (None, None):
                    file.write(f"{label} {x}, {y}\n")
        self._mark_dirty(False)
        try:
            update_registration_button()
        except NameError:
            pass

    # 저장된 좌표 txt 파일을 호출
    def load_coordinates_from_txt(self, file_name):
        with open(file_name, 'r+', encoding='utf-8') as file:
            for line in file:
                data = line.strip().split(' ')
                if len(data) == 3:
                    index, x, y = int(data[0]), float(data[1].replace(',','')), float(data[2].replace(',',''))
                    self.add_coordinate_img(index, x, y)
        self._mark_dirty(False)
        try:
            update_registration_button()
        except NameError:
            pass

    # 호출된 좌표 데이터 txt 파일 기반으로 이미지 작성
    def add_coordinate_img(self, index, x, y):
        cross_size = 7
        pen = QPen(QColor(255, 0, 0))
        pos = QPointF(x, y)
        cross_item1 = QGraphicsLineItem(pos.x() - cross_size / 2, pos.y(), pos.x() + cross_size / 2, pos.y())
        cross_item2 = QGraphicsLineItem(pos.x(), pos.y() - cross_size / 2, pos.x(), pos.y() + cross_size / 2)
        cross_item1.setPen(pen)
        cross_item2.setPen(pen)

        self.cross_items.append(cross_item1)
        self.cross_items.append(cross_item2)

        self.scene.addItem(cross_item1)
        self.scene.addItem(cross_item2)

        number_item = QGraphicsTextItem(str(index))
        number_item.setPos(pos.x() + cross_size, pos.y() - cross_size)
        number_item.setDefaultTextColor(QColor(Qt.GlobalColor.red))
        font = QFont()
        font.setPointSize(7)
        number_item.setFont(font)
        self.scene.addItem(number_item)
        self.number_items.append(number_item)
        self.number_count += 1
        self.coordinates.append(((pos.x()), (pos.y())))
        try:
            update_registration_button()
        except NameError:
            pass

    # 레이블 변경
    def modify_coordinate_label(self, index, new_label):
        if 0 <= index < len(self.number_items):
            try:
                new_label = int(new_label)
                self.number_count = max(self.number_count, new_label)  # number_count가 적어도 new_label만큼 커지도록 합니다.
                self.number_items[index].setPlainText(str(new_label))
                self._mark_dirty(True)
                try:
                    update_registration_button()
                except NameError:
                    pass
            except ValueError:
                QMessageBox.warning(self, '잘못된 입력', '레이블에는 정수 값을 입력하세요.')

def register_images(img1, img2, points1, points2, keys):
    if img1 is None or img2 is None:
        raise ValueError("Images not loaded")
    
    src_points = points2.astype(np.float32)
    dst_points = points1.astype(np.float32)
    
    transform_matrix, inliers = cv2.findHomography(
        src_points, dst_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.5,
        maxIters=2000,
        confidence=0.995
    )
    
    if transform_matrix is None:
        raise ValueError("Failed to estimate homography transform")
    
    transformed_img = cv2.warpPerspective(
        img2,
        transform_matrix,
        (img1.shape[1], img1.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    points2_reshaped = np.array(points2, dtype=np.float32).reshape(-1, 1, 2)
    registered_points2 = cv2.perspectiveTransform(points2_reshaped, transform_matrix)
    registered_points2 = registered_points2.reshape(-1, 2)
    
    return transformed_img, transform_matrix, registered_points2, inliers, keys

def draw_point_matches(overlay_img, points1, points2, inliers, keys=None):
    for i, ((x1, y1), (x2, y2), inlier) in enumerate(zip(points1, points2, inliers)):
        idx = keys[i] if keys is not None else i + 1
        dist = np.hypot(x1 - x2, y1 - y2)
        if inlier:
            color = (0, 255, 0)  # Inlier: 녹색
        else:
            color = (0, 0, 255)  # Outlier: 빨간색
        cv2.circle(overlay_img, (int(round(x2)), int(round(y2))), 3, color, -1)
        # 숫자 표시 (흰색)
        cv2.putText(
            overlay_img,
            str(idx),
            (int(round(x2)) + 5, int(round(y2)) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        # 점수(오차) 표시 (노란색)
        cv2.putText(
            overlay_img,
            f"{dist:.1f}",
            (int(round(x2)) + 4, int(round(y2)) + 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 255, 255),  # 노란색
            1,
            cv2.LINE_AA
        )

def find_counterpart_image(file_path):
    """neonsat_L1G ↔ google_ref 대응 이미지 경로를 반환. 못 찾으면 None.

    예) .../neonsat_google_tie_points/neonsat_L1G/neonsat_L1G_R001_C003.png
      ↔ .../neonsat_google_tie_points/google_ref/google_ref_R001_C003.tif

    두 폴더는 같은 부모(neonsat_google_tie_points) 아래의 형제 폴더이며,
    파일명의 타일 식별자(R###_C###)로 짝을 찾는다. 폴더마다 접두사와
    확장자가 다르므로(neonsat은 .png, google은 .tif) 식별자만으로 매칭한다.
    """
    normalized = file_path.replace('\\', '/')
    basename = os.path.basename(normalized)

    # 타일 식별자(R###_C###) 추출
    match = re.search(r'R\d+_C\d+', basename)
    if not match:
        return None
    tile = match.group(0)

    # 현재 파일이 위치한 폴더(neonsat_L1G 또는 google_ref)와 대응 폴더 결정
    src_dir = os.path.dirname(normalized)
    parent_dir = os.path.dirname(src_dir)
    src_folder = os.path.basename(src_dir)

    if src_folder == 'neonsat_L1G':
        target_folder = 'google_ref'
    elif src_folder == 'google_ref':
        target_folder = 'neonsat_L1G'
    else:
        return None

    # 대응 폴더에서 같은 타일 식별자를 가진 이미지 파일 탐색
    target_dir = f"{parent_dir}/{target_folder}"
    for ext in ImageViewer.IMAGE_EXTENSIONS:
        counterpart = f"{target_dir}/{target_folder}_{tile}{ext}"
        if os.path.exists(counterpart):
            # 원래 OS의 경로 구분자로 복원
            if '\\' in file_path:
                counterpart = counterpart.replace('/', '\\')
            return counterpart

    return None


def _create_lock_icon(locked):
    """자물쇠 아이콘 생성 (locked=True: 잠긴 자물쇠, locked=False: 열린 자물쇠)"""
    size = 32
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Shackle (U자형 고리)
    shackle_pen = QPen(QColor(80, 80, 80), 3)
    painter.setPen(shackle_pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    if locked:
        path.moveTo(10, 17)
        path.lineTo(10, 11)
        path.cubicTo(10, 4, 22, 4, 22, 11)
        path.lineTo(22, 17)
    else:
        path.moveTo(10, 17)
        path.lineTo(10, 11)
        path.cubicTo(10, 4, 22, 4, 22, 11)
        path.lineTo(22, 7)
    painter.drawPath(path)

    # Lock body (사각형 몸체)
    painter.setPen(QPen(QColor(80, 80, 80), 1.5))
    if locked:
        painter.setBrush(QColor(255, 193, 7))   # Gold
    else:
        painter.setBrush(QColor(180, 180, 180))  # Gray
    painter.drawRoundedRect(6, 16, 20, 13, 2, 2)

    # Keyhole (열쇠 구멍)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(80, 80, 80))
    painter.drawEllipse(QPointF(16, 21), 2, 2)
    painter.drawRect(15, 22, 2, 3)

    painter.end()
    return QIcon(pixmap)


class Image_Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.viewer = ImageViewer()
        self.setCentralWidget(self.viewer)
        self.folder_name = None
        self.current_image_path = None
        self.partner_window = None
        self._auto_loading = False
        self._sync_enabled = False
        self._syncing = False
        self.is_overlay = False
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Registration Tool 1")
        self.setWindowIcon(QIcon('./icon/earth.png'))
        self.move(0, 0)
        # 고정 크기 대신 초기 크기만 지정 → 사용자가 창 크기를 조절할 수 있음
        self.resize(1000, 1000)
        
        # 키보드 포커스 설정
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        self.create_toolbar()

    def update_title(self):
        if self.folder_name:
            title = self.folder_name
            if self.viewer._dirty:
                title += " *"
            self.setWindowTitle(title)

    def create_toolbar(self):
        toolbar = QToolBar("toolbar")
        self.addToolBar(toolbar)

        # 파일 불러오기
        open_Aciton = QAction(QIcon('./icon/open_img.png'), 'Open Image', self)
        open_Aciton.setStatusTip('Load Files')
        open_Aciton.triggered.connect(self.open_image)
        toolbar.addAction(open_Aciton)

        # 좌표 데이터 불러오기
        open_txt_Aciton = QAction(QIcon('./icon/open_img_txt.png'), 'Open Image with Coordinates', self)
        open_txt_Aciton.setStatusTip('Load Files')
        open_txt_Aciton.triggered.connect(self.open_image_with_coordinates)
        toolbar.addAction(open_txt_Aciton)

        # 이미지 확대
        plus_Action = QAction(QIcon('./icon/zoom_in.png'), 'Zoom In', self)
        plus_Action.setStatusTip('Zoom in Image')
        plus_Action.triggered.connect(self.zoom_in)
        toolbar.addAction(plus_Action)

        # 이미지 축소
        minus_Action = QAction(QIcon('./icon/zoom_out.png'), 'Zoom Out', self)
        minus_Action.setStatusTip('Zoom out Image')
        minus_Action.triggered.connect(self.zoom_out)
        toolbar.addAction(minus_Action)

        # Sync 토글
        self.sync_action = QAction(_create_lock_icon(False), 'Sync Views', self)
        self.sync_action.setStatusTip('Sync zoom and pan between windows')
        self.sync_action.setCheckable(True)
        self.sync_action.triggered.connect(self._toggle_sync)
        toolbar.addAction(self.sync_action)

        toolbar.addSeparator()

        # 이미지 저장
        save_img_Action = QAction(QIcon('./icon/save_img.png'), 'Save Image', self)
        save_img_Action.setStatusTip('Zoom out Image')
        save_img_Action.triggered.connect(self.save_coordinates_image)
        toolbar.addAction(save_img_Action)

        # 좌표 저장
        save_txt_Action = QAction(QIcon('./icon/save_txt.png'), 'Save Coordinates', self)
        save_txt_Action.setStatusTip('Save Coordinate to txt')
        save_txt_Action.triggered.connect(self.save_coordinate_txt)
        toolbar.addAction(save_txt_Action)

        # 좌표 전체 삭제
        all_erase_Action = QAction(QIcon('./icon/erase.png'), 'All Erase', self)
        all_erase_Action.setStatusTip('All Erase coordinate')
        all_erase_Action.triggered.connect(self.confirm_clear_all_coordinates)
        toolbar.addAction(all_erase_Action)

        # 나가기
        exit_Action = QAction(QIcon('./icon/exit.png'), 'Exit', self)
        exit_Action.setStatusTip('Exit application')
        exit_Action.triggered.connect(self.confirm_exit_application)
        toolbar.addAction(exit_Action)

        self.statusBar()

    def open_image(self, file_name=None):
        if not file_name:
            options = QFileDialog.Option.DontUseNativeDialog
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp *.gif *.tif);;All Files (*)", options=options)

        if file_name:
            if self.viewer._dirty:
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Icon.Warning)
                msg.setWindowTitle('저장 확인')
                msg.setText('현재 좌표가 저장되지 않았습니다. 저장하시겠습니까?')
                msg.setStandardButtons(
                    QMessageBox.StandardButton.Save |
                    QMessageBox.StandardButton.Discard |
                    QMessageBox.StandardButton.Cancel
                )
                msg.setDefaultButton(QMessageBox.StandardButton.Save)
                reply = msg.exec()
                if reply == QMessageBox.StandardButton.Save:
                    if not self.save_coordinate_txt():
                        # 저장이 무결성 오류/취소로 막히면 새 이미지를 열지 않음
                        return
                elif reply == QMessageBox.StandardButton.Cancel:
                    return

            self.current_image_path = file_name
            basename = os.path.basename(file_name)
            folder_name, _ = os.path.splitext(basename)
            self.folder_name = folder_name
            self.update_title()
            self.viewer.undo_stack.clear()
            self.viewer.load_image(file_name)

    def save_coordinates_image(self):
        dialog = QFileDialog(self, "Save Image with Coordinates", f"{self.folder_name}")
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilter("Images (*.png *.jpg *.bmp);;All Files (*)")
        dialog.setDefaultSuffix("png")
        dialog.setOptions(QFileDialog.Option.DontUseNativeDialog)
        
        if dialog.exec():
            file_name = dialog.selectedFiles()[0]
            # 뷰의 현재 변환 상태에 영향을 받지 않고 전체 씬을 렌더링하여 정확한 위치에 좌표가 저장되도록 합니다.
            scene_rect = self.viewer.sceneRect()
            pixmap = QPixmap(scene_rect.size().toSize())
            pixmap.fill(Qt.GlobalColor.transparent)

            painter = QPainter(pixmap)
            # 씬의 특정 영역(scene_rect)을 QPixmap의 특정 영역(pixmap.rect())에 렌더링합니다.
            self.viewer.scene.render(painter, QRectF(pixmap.rect()), scene_rect)
            painter.end()
            
            pixmap.save(file_name)
            
    def save_coordinate_txt(self):
        ok, message = self.check_pair_integrity()
        if not ok:
            self._show_integrity_warning(message)
            return False

        dialog = QFileDialog(self, "Save txt with Coordinates", f"{self.folder_name}")
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilter(".txt (*.txt);;All Files (*)")
        dialog.setDefaultSuffix("txt")
        dialog.setOptions(QFileDialog.Option.DontUseNativeDialog)

        if dialog.exec():
            file_name = dialog.selectedFiles()[0]
            if file_name:
                self.viewer.save_coordinates_to_txt(file_name)
                return True
        return False

    def zoom_in(self):
        self.viewer.plus_image()

    def zoom_out(self):
        self.viewer.minus_image()

    def open_image_auto(self, file_name):
        """이미지를 열고, 같은 이름의 .txt 좌표 파일이 있으면 자동으로 함께 로드"""
        # 미저장 좌표가 있으면 저장 여부 확인
        if self.viewer._dirty:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle('저장 확인')
            msg.setText('현재 좌표가 저장되지 않았습니다. 저장하시겠습니까?')
            msg.setStandardButtons(
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel
            )
            msg.setDefaultButton(QMessageBox.StandardButton.Save)
            reply = msg.exec()

            if reply == QMessageBox.StandardButton.Save:
                if not self.save_coordinate_txt():
                    # 저장이 무결성 오류/취소로 막히면 새 이미지를 열지 않음
                    return
            elif reply == QMessageBox.StandardButton.Cancel:
                return

        self.current_image_path = file_name
        basename = os.path.basename(file_name)
        folder_name, _ = os.path.splitext(basename)
        self.folder_name = folder_name
        self.update_title()
        self.viewer.undo_stack.clear()
        self.viewer.load_image(file_name)

        txt_file = os.path.splitext(file_name)[0] + '.txt'
        if os.path.exists(txt_file):
            self.viewer.load_coordinates_from_txt(txt_file)

        # NEONSAT/Google 대응 이미지 자동 로딩
        if not self._auto_loading and self.partner_window:
            counterpart = find_counterpart_image(file_name)
            if counterpart:
                self._auto_loading = True
                try:
                    self.partner_window.open_image_auto(counterpart)
                finally:
                    self._auto_loading = False

    def _toggle_sync(self, checked):
        """Sync 토글: 양쪽 윈도우의 줌/패닝 연동"""
        self._sync_enabled = checked
        self.sync_action.setIcon(_create_lock_icon(checked))

        if self.partner_window:
            self.partner_window._sync_enabled = checked
            self.partner_window.sync_action.blockSignals(True)
            self.partner_window.sync_action.setChecked(checked)
            self.partner_window.sync_action.setIcon(_create_lock_icon(checked))
            self.partner_window.sync_action.blockSignals(False)

        if checked:
            self._sync_to_partner()

    def _sync_to_partner(self):
        """현재 뷰의 줌/스크롤 상태를 파트너 윈도우에 동기화"""
        if not self._sync_enabled or self._syncing or not self.partner_window:
            return

        self._syncing = True
        try:
            partner = self.partner_window
            partner.viewer.setTransform(self.viewer.transform())
            center = self.viewer.mapToScene(self.viewer.viewport().rect().center())
            partner.viewer.centerOn(center)
        finally:
            self._syncing = False

    def open_image_with_coordinates(self):
        options = QFileDialog.Option.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "이미지 파일 열기", "", "이미지 (*.png *.jpg *.bmp *.gif *.tif);;모든 파일 (*)", options=options)
        if file_name:
            self.current_image_path = file_name
            basename = os.path.basename(file_name)
            folder_name, _ = os.path.splitext(basename)
            self.folder_name = folder_name

            self.viewer.load_image(file_name)
            self.update_title()
            # 기존 좌표 데이터 로드
            coord_file, _ = QFileDialog.getOpenFileName(self, "좌표 파일 열기", "", ".txt (*.txt);;모든 파일 (*)", options=options)
            if coord_file:
                self.viewer.load_coordinates_from_txt(coord_file)
    
    def confirm_clear_all_coordinates(self):
        reply = QMessageBox.question(self, '좌표 삭제 확인', 
                                   '모든 좌표를 삭제하시겠습니까?',
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                   QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.viewer.remove_cross_items()
    
    def confirm_exit_application(self):
        if self.viewer._dirty:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle('종료 확인')
            msg.setText('저장되지 않은 좌표가 있습니다. 저장하시겠습니까?')
            msg.setStandardButtons(
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel
            )
            msg.setDefaultButton(QMessageBox.StandardButton.Save)
            reply = msg.exec()
            if reply == QMessageBox.StandardButton.Save:
                if self.quick_save_coordinates():
                    QApplication.instance().quit()
                # 저장이 막히면 종료하지 않고 무결성 문제 해결을 유도
            elif reply == QMessageBox.StandardButton.Discard:
                QApplication.instance().quit()
        else:
            reply = QMessageBox.question(self, '종료 확인',
                                       '프로그램을 종료하시겠습니까?',
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                       QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                QApplication.instance().quit()
    
    @staticmethod
    def _index_stats(viewer):
        """뷰어의 좌표 레이블을 정수 인덱스로 집계.

        반환: (인덱스 집합, 중복 인덱스 리스트, 정수가 아닌 레이블 리스트)
        """
        counts = {}
        non_int = []
        for item in viewer.number_items:
            label = item.toPlainText()
            try:
                value = int(label)
            except ValueError:
                non_int.append(label)
                continue
            counts[value] = counts.get(value, 0) + 1
        indices = set(counts)
        dups = sorted(v for v, c in counts.items() if c > 1)
        return indices, dups, non_int

    def check_pair_integrity(self):
        """두 윈도우의 정합점 무결성 검사.

        - 두 윈도우의 정합점 개수가 일치하는가
        - 한 윈도우 안에 중복되는 인덱스가 없는가
        - 인덱스가 1:1로 매핑되는가(짝 없이 남는 인덱스가 없는가)

        문제 없으면 (True, ''), 있으면 (False, 문제 설명 메시지)를 반환한다.
        """
        partner = self.partner_window
        if partner is None:
            return True, ''

        name_a = self.folder_name or self.windowTitle()
        name_b = partner.folder_name or partner.windowTitle()

        idx_a, dup_a, bad_a = self._index_stats(self.viewer)
        idx_b, dup_b, bad_b = self._index_stats(partner.viewer)

        problems = []
        if bad_a:
            problems.append(f"[{name_a}] 정수가 아닌 레이블: {', '.join(bad_a)}")
        if bad_b:
            problems.append(f"[{name_b}] 정수가 아닌 레이블: {', '.join(bad_b)}")
        if dup_a:
            problems.append(f"[{name_a}] 중복된 인덱스: {', '.join(map(str, dup_a))}")
        if dup_b:
            problems.append(f"[{name_b}] 중복된 인덱스: {', '.join(map(str, dup_b))}")

        only_a = sorted(idx_a - idx_b)
        only_b = sorted(idx_b - idx_a)
        if only_a:
            problems.append(f"[{name_a}]에만 있어 짝이 없는 인덱스: {', '.join(map(str, only_a))}")
        if only_b:
            problems.append(f"[{name_b}]에만 있어 짝이 없는 인덱스: {', '.join(map(str, only_b))}")

        count_a = len(self.viewer.coordinates)
        count_b = len(partner.viewer.coordinates)
        if count_a != count_b:
            problems.append(f"정합점 개수 불일치: [{name_a}] {count_a}개 ↔ [{name_b}] {count_b}개")

        if problems:
            return False, '\n'.join(problems)
        return True, ''

    def _show_integrity_warning(self, message):
        QMessageBox.warning(
            self, '정합점 무결성 오류',
            '정합점 쌍에 문제가 있어 저장할 수 없습니다.\n'
            '아래 문제를 해결한 뒤 다시 저장하세요.\n\n' + message
        )

    def quick_save_coordinates(self):
        """Ctrl+S: 이미지와 같은 디렉토리에 동일 이름의 .txt로 좌표 즉시 저장"""
        if not self.current_image_path or not self.viewer.coordinates:
            return True
        ok, message = self.check_pair_integrity()
        if not ok:
            self._show_integrity_warning(message)
            return False
        txt_path = os.path.splitext(self.current_image_path)[0] + '.txt'
        self.viewer.save_coordinates_to_txt(txt_path)
        self.statusBar().showMessage(f'저장 완료: {txt_path}', 3000)
        return True

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_S and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.quick_save_coordinates()
        elif event.key() == Qt.Key.Key_Z and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.viewer.undo()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        # 정합 결과(오버레이) 창은 하나의 결과 뷰일 뿐이므로,
        # 프로그램 종료 확인 없이 그냥 닫는다.
        if self.is_overlay:
            event.accept()
            return
        if self.viewer._dirty:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle('종료 확인')
            msg.setText('저장되지 않은 좌표가 있습니다. 저장하시겠습니까?')
            msg.setStandardButtons(
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel
            )
            msg.setDefaultButton(QMessageBox.StandardButton.Save)
            reply = msg.exec()
            if reply == QMessageBox.StandardButton.Save:
                if self.quick_save_coordinates():
                    event.accept()
                else:
                    # 저장이 막히면 종료하지 않고 무결성 문제 해결을 유도
                    event.ignore()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            reply = QMessageBox.question(self, '종료 확인',
                                       '프로그램을 종료하시겠습니까?',
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                       QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                event.accept()
            else:
                event.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    Window_one = Image_Window()
    Window_one.show()

    Window_two = Image_Window()
    Window_two.setWindowTitle("Image Registration Tool 2")
    Window_two.move(350, 0)

    Window_one.partner_window = Window_two
    Window_two.partner_window = Window_one
    Window_two.show()

    active_overlays = []

    def get_common_count():
        try:
            points1_dict = {int(n.toPlainText()): list(c) for n, c in zip(Window_one.viewer.number_items, Window_one.viewer.coordinates)}
            points2_dict = {int(n.toPlainText()): list(c) for n, c in zip(Window_two.viewer.number_items, Window_two.viewer.coordinates)}
            return len(set(points1_dict) & set(points2_dict))
        except:
            return 0

    def update_registration_button():
        reg_action.setEnabled(get_common_count() >= 5)

    # Add registration button to Window_two's toolbar
    toolbar = Window_two.findChild(QToolBar)
    reg_action = QAction(QIcon('./icon/reg_img.png'), "Perform Registration", Window_two)
    reg_action.setStatusTip("Register images and show overlay")
    toolbar.addAction(reg_action)

    def registration_func():
        try:
            if Window_one.viewer.image_item is None or Window_two.viewer.image_item is None:
                QMessageBox.warning(Window_two, "Error", "Both windows must have images loaded.")
                return

            # 정합 전 정합점 무결성 검사 (개수 일치, 중복 없음, 1:1 매핑)
            ok, message = Window_two.check_pair_integrity()
            if not ok:
                Window_two._show_integrity_warning(message)
                return

            points1_dict = {int(n.toPlainText()): [c[0], c[1]] for n, c in zip(Window_one.viewer.number_items, Window_one.viewer.coordinates)}
            points2_dict = {int(n.toPlainText()): [c[0], c[1]] for n, c in zip(Window_two.viewer.number_items, Window_two.viewer.coordinates)}
            common_keys = sorted(set(points1_dict) & set(points2_dict))
            if len(common_keys) < 5:
                QMessageBox.warning(Window_two, "Error", "At least 5 common points required.")
                return

            points1 = np.array([points1_dict[k] for k in common_keys])
            points2 = np.array([points2_dict[k] for k in common_keys])

            def pixmap_to_cv(pixmap):
                image = pixmap.toImage()
                width = image.width()
                height = image.height()
                bits = image.constBits()
                bits.setsize(height * width * 4)
                arr = np.frombuffer(bits, np.uint8).reshape((height, width, 4))
                return arr[:, :, [0, 1, 2]]  # BGRA to BGR

            img1 = pixmap_to_cv(Window_one.viewer.image_item.pixmap())
            img2 = pixmap_to_cv(Window_two.viewer.image_item.pixmap())

            transformed_img, _, registered_points2, inliers, _ = register_images(img1, img2, points1, points2, common_keys)

            overlay_img = cv2.addWeighted(img1, 0.5, transformed_img, 0.5, 0)
            draw_point_matches(overlay_img, points1, registered_points2, inliers.ravel(), common_keys)

            overlay_window = Image_Window()
            overlay_window.is_overlay = True
            overlay_window.setWindowTitle("Overlay Registration Result")
            overlay_window.viewer.load_from_numpy(overlay_img)
            
            # ----- Add Save Overlay button -----
            overlay_toolbar = overlay_window.findChild(QToolBar)
            save_overlay_action = QAction(QIcon('./icon/save.png'), 'Save Overlay', overlay_window)
            save_overlay_action.setStatusTip('Save overlay image')
            
            def save_overlay():
                # Default filename: original Window_two image name + '_overlay.png'
                default_name = f"{Window_two.folder_name}_overlay.png" if Window_two.folder_name else 'overlay.png'
                dialog = QFileDialog(overlay_window, 'Save Overlay Image', default_name)
                dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
                dialog.setNameFilter('Images (*.png *.jpg *.bmp);;All Files (*)')
                dialog.setDefaultSuffix('png')
                dialog.setOptions(QFileDialog.Option.DontUseNativeDialog)
                if dialog.exec():
                    file_name = dialog.selectedFiles()[0]
                    # Get pixmap from viewer and save
                    pixmap = overlay_window.viewer.image_item.pixmap()
                    pixmap.save(file_name)
            
            save_overlay_action.triggered.connect(save_overlay)
            if overlay_toolbar is not None:
                overlay_toolbar.addAction(save_overlay_action)
            else:
                # If no toolbar found (shouldn't happen), create one
                new_toolbar = QToolBar("overlay_toolbar", overlay_window)
                overlay_window.addToolBar(new_toolbar)
                new_toolbar.addAction(save_overlay_action)
            # ----- End Save Overlay button -----
            
            overlay_window.show()
            active_overlays.append(overlay_window)
        except Exception as e:
            QMessageBox.critical(Window_two, "Error", str(e))

    reg_action.triggered.connect(registration_func)
    update_registration_button()

    sys.exit(app.exec())

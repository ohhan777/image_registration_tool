import sys
import typing
import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsLineItem, QGraphicsTextItem,
    QToolBar, QFileDialog, QInputDialog, QLineEdit, QMessageBox
)
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QIcon, QAction, QKeySequence
from PIL import Image, ImageDraw, ImageFont
import os

import cv2
from PyQt6.QtGui import QImage

class ImageViewer(QGraphicsView):
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

    def init_ui(self):
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.image_item = None
        self.cross_items = []
        self.number_items = []
        self.coordinates = []
        self.number_count = 0

        self.zoom_factor = 1.0
        self.zoom_step = 0.1
        self.min_zoom = 0.05
        self.max_zoom = 2.0
        self.setMinimumSize(400, 400)
        
        # 드래그 스크롤 관련 변수
        self.last_pan_point = None
        self.is_panning = False
        
        # Undo 관련 변수
        self.undo_stack = []

    def load_image(self, file_name):
        self.scene.clear()
        self.cross_items.clear()
        self.number_items.clear()
        self.coordinates.clear()
        self.number_count = 0
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

    def minus_image(self):
        if self.zoom_factor > self.zoom_step:
            self.zoom_factor -= self.zoom_step
            self.scale(self.zoom_factor, self.zoom_factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Ctrl 키가 눌려있으면 패닝 모드로 전환
            if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                self.last_pan_point = event.position().toPoint()
                self.is_panning = True
                return
                
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
        elif event.button() == Qt.MouseButton.RightButton:
            pos = self.mapToScene(event.position().toPoint())
            # 클릭한 좌표 주변에 있는 좌표를 삭제
            if self.remove_coordinates(pos):
                return
        elif event.button() == Qt.MouseButton.MiddleButton:
            # 중간 마우스 버튼으로 패닝 시작
            self.last_pan_point = event.position().toPoint()
            self.is_panning = True

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
        try:
            update_registration_button()
        except NameError:
            pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Z and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.undo()
        else:
            super().keyPressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_panning and self.last_pan_point is not None:
            # 드래그 거리 계산
            delta = event.position().toPoint() - self.last_pan_point
            self.last_pan_point = event.position().toPoint()
            
            # 스크롤바 이동
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = False
            self.last_pan_point = None
        elif event.button() == Qt.MouseButton.LeftButton and self.is_panning:
            # Ctrl+왼쪽 클릭 드래그 종료
            self.is_panning = False
            self.last_pan_point = None
        
        super().mouseReleaseEvent(event)

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

    # txt 파일로 좌표 데이터 저장
    def save_coordinates_to_txt(self, file_name):
        if not self.coordinates:
            return 
        
        with open(file_name, 'w') as file:
            for number_item, (x, y) in zip(self.number_items, self.coordinates):
                label = number_item.toPlainText()
                if (x, y) != (None, None):
                    file.write(f"{label} {x}, {y}\n")
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

class Image_Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.viewer = ImageViewer()
        self.setCentralWidget(self.viewer)
        self.folder_name = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Registration Tool 1")
        self.setWindowIcon(QIcon('./icon/earth.png'))
        self.move(0, 0)
        self.setFixedSize(1000, 1000)
        
        # 키보드 포커스 설정
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        self.create_toolbar()

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
        all_erase_Action = QAction(QIcon('./icon/erase.png'), 'all_Erase', self)
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
            basename = os.path.basename(file_name)
            folder_name, _ = os.path.splitext(basename)
            self.folder_name = folder_name
            self.setWindowTitle(self.folder_name)
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
        dialog = QFileDialog(self, "Save txt with Coordinates", f"{self.folder_name}")
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilter(".txt (*.txt);;All Files (*)")
        dialog.setDefaultSuffix("txt")
        dialog.setOptions(QFileDialog.Option.DontUseNativeDialog)

        if dialog.exec():
            file_name = dialog.selectedFiles()[0]
            if file_name:
                self.viewer.save_coordinates_to_txt(file_name)

    def zoom_in(self):
        self.viewer.plus_image()

    def zoom_out(self):
        self.viewer.minus_image()

    def open_image_with_coordinates(self):
        options = QFileDialog.Option.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "이미지 파일 열기", "", "이미지 (*.png *.jpg *.bmp *.gif *.tif);;모든 파일 (*)", options=options)
        if file_name:
            basename = os.path.basename(file_name)
            folder_name, _ = os.path.splitext(basename)
            self.folder_name = folder_name

            self.viewer.load_image(file_name)
            self.setWindowTitle(self.folder_name)
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
        reply = QMessageBox.question(self, '종료 확인', 
                                   '프로그램을 종료하시겠습니까?',
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                   QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            QApplication.instance().quit()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Z and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.viewer.undo()
        else:
            super().keyPressEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    Window_one = Image_Window()
    Window_one.show()

    Window_two = Image_Window()
    Window_two.setWindowTitle("Image Registration Tool 2")
    Window_two.move(350, 0)
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

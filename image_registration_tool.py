import sys
import typing
import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsLineItem, QGraphicsTextItem,
    QToolBar, QFileDialog, QInputDialog, QLineEdit, QMessageBox
)
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QIcon, QAction
from PIL import Image, ImageDraw, ImageFont
import os

class ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.init_ui()

        # 이미지의 중심을 중심으로 확대/축소하도록 설정
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        # 마우스 휠 이벤트 핸들링
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

    def init_ui(self):
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.image_item = None
        self.cross_items = []
        self.number_items = []
        self.coordinates = []

        self.zoom_factor = 1.0
        self.zoom_step = 0.1
        self.min_zoom = 0.05
        self.max_zoom = 2.0
        self.setMinimumSize(400, 400)

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

    def plus_image(self):
        self.zoom_factor += self.zoom_step
        self.scale(self.zoom_factor, self.zoom_factor)

    def minus_image(self):
        if self.zoom_factor > self.zoom_step:
            self.zoom_factor -= self.zoom_step
            self.scale(self.zoom_factor, self.zoom_factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
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

    def Click_Coordinate(self, pos):
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

    def remove_cross_one_item(self, index):
        if 0 <= index < len(self.cross_items) and 0 <= index < len(self.number_items):
            cross_item1 = self.cross_items.pop(index * 2)
            cross_item2 = self.cross_items.pop(index * 2)
            self.scene.removeItem(cross_item1)
            self.scene.removeItem(cross_item2)
            self.scene.removeItem(self.number_items[index])
            self.number_items.pop(index)
            self.number_count -= 1

    # 좌표 개별 삭제
    def remove_coordinates(self, pos):
        for i, (x, y) in enumerate(self.coordinates):
            distance = (x - pos.x())**2 + (y - pos.y())**2
            if distance < 9:
                self.remove_cross_one_item(i)
                self.coordinates.pop(i)
                return True
        return False

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
            for i, _ in enumerate(self.number_items):
                x, y = self.coordinates[i]  
                if (x, y) != (None, None):
                    file.write(f"{i+1} {x}, {y}\n")

    # 저장된 좌표 txt 파일을 호출
    def load_coordinates_from_txt(self, file_name):
        with open(file_name, 'r+', encoding='utf-8') as file:
            for line in file:
                data = line.strip().split(' ')
                if len(data) == 3:
                    index, x, y = int(data[0]), float(data[1].replace(',','')), float(data[2].replace(',',''))
                    self.add_coordinate_img(index, x, y)

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

    # 레이블 변경
    def modify_coordinate_label(self, index, new_label):
        if 0 <= index < len(self.number_items):
            try:
                new_label = int(new_label)
                self.number_count = max(self.number_count, new_label)  # number_count가 적어도 new_label만큼 커지도록 합니다.
                self.number_items[index].setPlainText(str(new_label))
            except ValueError:
                QMessageBox.warning(self, '잘못된 입력', '레이블에는 정수 값을 입력하세요.')

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

        self.create_toolbar()

    def create_toolbar(self):
        toolbar = QToolBar("toolbar")
        self.addToolBar(toolbar)

        # 파일 불러오기
        open_Aciton = QAction(QIcon('./icon/open.png'), 'Open', self)
        open_Aciton.setStatusTip('Load Files')
        open_Aciton.triggered.connect(self.open_image)
        toolbar.addAction(open_Aciton)

        # 좌표 데이터 불러오기
        open_txt_Aciton = QAction(QIcon('./icon/coor_on_img.png'), 'Open', self)
        open_txt_Aciton.setStatusTip('Load Files')
        open_txt_Aciton.triggered.connect(self.open_image_with_coordinates)
        toolbar.addAction(open_txt_Aciton)

        # 이미지 확대
        plus_Action = QAction(QIcon('./icon/plus.png'), 'Zoom in', self)
        plus_Action.setStatusTip('Zoom in Image')
        plus_Action.triggered.connect(self.zoom_in)
        toolbar.addAction(plus_Action)

        # 이미지 축소
        minus_Action = QAction(QIcon('./icon/minus.png'), 'Zoom out', self)
        minus_Action.setStatusTip('Zoom out Image')
        minus_Action.triggered.connect(self.zoom_out)
        toolbar.addAction(minus_Action)

        # 이미지 저장
        save_img_Action = QAction(QIcon('./icon/save.png'), 'Save', self)
        save_img_Action.setStatusTip('Zoom out Image')
        save_img_Action.triggered.connect(self.save_coordinates_image)
        toolbar.addAction(save_img_Action)

        # 좌표 저장
        save_txt_Action = QAction(QIcon('./icon/txt_coordinate.png'), 'txt_coordinate', self)
        save_txt_Action.setStatusTip('Save Coordinate to txt')
        save_txt_Action.triggered.connect(self.save_coordinate_txt)
        toolbar.addAction(save_txt_Action)

        # 좌표 전체 삭제
        all_erase_Action = QAction(QIcon('./icon/all_erase.png'), 'all_Erase', self)
        all_erase_Action.setStatusTip('All Erase coordinate')
        all_erase_Action.triggered.connect(self.viewer.remove_cross_items)
        toolbar.addAction(all_erase_Action)

        # 나가기
        exit_Action = QAction(QIcon('./icon/exit.png'), 'Exit', self)
        exit_Action.setStatusTip('Exit application')
        exit_Action.triggered.connect(QApplication.instance().quit)
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

if __name__ == '__main__':
    app = QApplication(sys.argv)

    Window_one = Image_Window()
    Window_one.show()

    Window_two = Image_Window()
    Window_two.setWindowTitle("Image Registration Tool 2")
    Window_two.move(350, 0)
    Window_two.show()

    # Window_three = Image_Window()
    # Window_three.setWindowTitle("Image Registration Tool_3")
    # Window_three.move(700, 0)
    # Window_three.show()

    # Window_four = Image_Window()
    # Window_four.setWindowTitle("Image Registration Tool_4")
    # Window_four.move(1050, 0)
    # Window_four.show()

    sys.exit(app.exec())
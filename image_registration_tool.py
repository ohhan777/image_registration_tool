import os
# GeoTIFF(google_ref .tif)의 지리 메타데이터 태그를 Qt TIFF 플러그인이
# 인식하지 못해 찍는 "Unknown field with tag ..." 경고를 끈다.
# (이미지 로드에는 문제가 없으며 콘솔 경고만 억제)
os.environ.setdefault("QT_LOGGING_RULES", "qt.imageformats.tiff=false")

import sys
import typing
import contextlib
import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF, QSettings, QTimer, QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem,
    QToolBar, QFileDialog, QInputDialog, QLineEdit, QMessageBox,
    QSlider, QLabel, QWidget, QSizePolicy, QPushButton, QDoubleSpinBox,
    QDialog, QFormLayout, QHBoxLayout, QSpinBox, QColorDialog
)
from PyQt6.QtGui import (
    QPixmap, QPainter, QPen, QColor, QFont, QFontMetricsF, QIcon, QAction, QKeySequence, QPainterPath
)
from PIL import Image, ImageDraw, ImageFont
import re

import cv2
from PyQt6.QtGui import QImage

# 밝기/대비 값을 창별로 기억해 두는 설정 저장소 (재실행 시 복원)
SETTINGS_ORG = 'KARI'
SETTINGS_APP = 'ImageRegistrationTool'


def _build_adjust_lut(brightness, contrast):
    """밝기/대비 슬라이더 값(-100~100)을 8비트 룩업테이블로 만든다.

    대비는 128을 중심으로 기울기를 바꾸고, 밝기는 그 뒤에 오프셋을 더한다.
    둘 다 0이면 항등 변환이다. 슬라이더 한 칸이 DN 2.55에 해당하므로
    ±100에서 전체 범위(±255)를 덮는다.
    """
    c = contrast * 2.55
    factor = (259.0 * (c + 255.0)) / (255.0 * (259.0 - c))
    x = np.arange(256, dtype=np.float32)
    y = factor * (x - 128.0) + 128.0 + brightness * 2.55
    return np.clip(y, 0, 255).astype(np.uint8)


class MarkerStyle(QObject):
    """정합점 마커/번호의 색과 크기. 모든 창이 하나의 설정을 공유한다.

    값이 바뀌면 changed 신호로 알려서 열려 있는 모든 뷰가 다시 그려지게 하고,
    QSettings에 저장해 다음 실행에도 그대로 유지한다.
    """

    changed = pyqtSignal()

    DEFAULT_COLORS = {
        'point': '#ff0000',    # 원본 창에 찍는 정합점
        'inlier': '#00ff00',   # 정합 결과: RANSAC inlier
        'outlier': '#ff3c3c',  # 정합 결과: RANSAC outlier
        'text': '#ffff00',     # 정합 결과의 번호/잔차 글자
    }
    DEFAULT_MARKER_SIZE = 13
    DEFAULT_FONT_SIZE = 12
    MARKER_SIZE_RANGE = (5, 60)
    FONT_SIZE_RANGE = (6, 40)

    def __init__(self):
        super().__init__()
        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self._colors = {}
        for role, default in self.DEFAULT_COLORS.items():
            color = QColor(str(settings.value(f'markers/{role}', default)))
            self._colors[role] = color if color.isValid() else QColor(default)
        self.marker_size = self._stored_int(settings, 'markers/size',
                                            self.DEFAULT_MARKER_SIZE, self.MARKER_SIZE_RANGE)
        self.font_size = self._stored_int(settings, 'markers/font',
                                          self.DEFAULT_FONT_SIZE, self.FONT_SIZE_RANGE)

    @staticmethod
    def _stored_int(settings, key, default, limits):
        """설정 파일이 손상돼도 실행을 막지 않도록 범위를 벗어난 값은 잘라낸다."""
        try:
            value = int(settings.value(key, default))
        except (TypeError, ValueError):
            return default
        return max(limits[0], min(limits[1], value))

    @property
    def line_width(self):
        """마커 크기에 비례하는 선 두께 (화면 픽셀)."""
        return max(1.0, round(self.marker_size / 7.0))

    def color(self, role):
        return self._colors.get(role, self._colors['point'])

    def set_color(self, role, color):
        if not color.isValid() or self._colors.get(role) == color:
            return
        self._colors[role] = color
        self._save(f'markers/{role}', color.name())
        self.changed.emit()

    def set_sizes(self, marker_size, font_size):
        if (marker_size, font_size) == (self.marker_size, self.font_size):
            return
        self.marker_size = marker_size
        self.font_size = font_size
        self._save('markers/size', marker_size)
        self._save('markers/font', font_size)
        self.changed.emit()

    def reset(self):
        self._colors = {role: QColor(value) for role, value in self.DEFAULT_COLORS.items()}
        self.marker_size = self.DEFAULT_MARKER_SIZE
        self.font_size = self.DEFAULT_FONT_SIZE
        for role, color in self._colors.items():
            self._save(f'markers/{role}', color.name())
        self._save('markers/size', self.marker_size)
        self._save('markers/font', self.font_size)
        self.changed.emit()

    @staticmethod
    def _save(key, value):
        QSettings(SETTINGS_ORG, SETTINGS_APP).setValue(key, value)


_marker_style = None


def marker_style():
    """공용 MarkerStyle 인스턴스 (QApplication 생성 이후 처음 쓸 때 만든다)."""
    global _marker_style
    if _marker_style is None:
        _marker_style = MarkerStyle()
    return _marker_style


class PointMarkerItem(QGraphicsItem):
    """줌 배율과 무관하게 항상 같은 화면 크기로 보이는 정합점 마커.

    ItemIgnoresTransformations를 켜면 아이템 로컬 좌표가 곧 화면 픽셀이 된다.
    그래서 pos()에는 씬(영상 픽셀) 좌표를 주고, 십자/원과 글자는 로컬에
    픽셀 단위로 그린다. 축소해도 마커가 같이 작아지지 않는다.
    """

    def __init__(self, label='', sub_label='', role='point', text_role=None, shape='cross'):
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setZValue(10)   # 항상 영상 위에
        self._label = label
        self._sub_label = sub_label
        self._role = role
        self._text_role = text_role or role
        self._shape = shape

    # 기존 QGraphicsTextItem 자리를 대신하므로 레이블 API 이름을 맞춘다
    def toPlainText(self):
        return self._label

    def setPlainText(self, text):
        self._label = str(text)
        self.style_changed()

    def style_changed(self):
        """색/크기 설정이 바뀌었을 때 크기 재계산과 다시 그리기를 요청한다."""
        self.prepareGeometryChange()
        self.update()

    def _font(self):
        font = QFont()
        # 포인트가 아닌 픽셀 단위 → 화면에서 항상 같은 크기로 보인다
        font.setPixelSize(marker_style().font_size)
        font.setBold(True)
        return font

    def _text_lines(self):
        return [line for line in (self._label, self._sub_label) if line]

    def _text_metrics(self):
        """(글꼴 정보, 텍스트 블록 너비, 높이)."""
        metrics = QFontMetricsF(self._font())
        lines = self._text_lines()
        if not lines:
            return metrics, 0.0, 0.0
        width = max(metrics.horizontalAdvance(line) for line in lines)
        return metrics, width, metrics.height() * len(lines)

    def boundingRect(self):
        style = marker_style()
        radius = style.marker_size / 2.0 + style.line_width
        _, text_width, text_height = self._text_metrics()
        left = -radius - 1.0
        top = min(-radius, -text_height / 2.0) - 1.0
        right = radius + (text_width + 8.0 if text_width else 0.0)
        bottom = max(radius, text_height / 2.0) + 1.0
        return QRectF(left, top, right - left, bottom - top)

    def paint(self, painter, option, widget=None):
        style = marker_style()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        radius = style.marker_size / 2.0
        painter.setPen(QPen(style.color(self._role), style.line_width))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if self._shape == 'circle':
            painter.drawEllipse(QPointF(0.0, 0.0), radius, radius)
        else:
            # 가운데를 비워 찍은 지점의 화소가 가려지지 않게 한다
            gap = max(1.0, radius * 0.35)
            painter.drawLine(QPointF(-radius, 0.0), QPointF(-gap, 0.0))
            painter.drawLine(QPointF(gap, 0.0), QPointF(radius, 0.0))
            painter.drawLine(QPointF(0.0, -radius), QPointF(0.0, -gap))
            painter.drawLine(QPointF(0.0, gap), QPointF(0.0, radius))

        lines = self._text_lines()
        if not lines:
            return

        metrics, text_width, text_height = self._text_metrics()
        left = radius + 4.0
        top = -text_height / 2.0

        # 밝은 영상 위에서도 글자가 묻히지 않도록 옅은 어두운 배경을 깐다
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 110))
        painter.drawRoundedRect(QRectF(left - 2.0, top, text_width + 4.0, text_height), 2.0, 2.0)

        painter.setPen(QPen(style.color(self._text_role)))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setFont(self._font())
        for i, line in enumerate(lines):
            painter.drawText(QPointF(left, top + metrics.ascent() + i * metrics.height()), line)


class MarkerStyleDialog(QDialog):
    """정합점 마커의 색/크기 설정 창.

    모덜리스로 띄워서 슬라이더를 만지는 즉시 결과를 확인할 수 있게 한다.
    """

    def __init__(self, parent, roles):
        super().__init__(parent)
        self.setWindowTitle('정합점 표시 설정')
        style = marker_style()

        layout = QFormLayout(self)
        self._color_buttons = {}
        for role, label in roles:
            button = QPushButton()
            button.setFixedWidth(90)
            button.clicked.connect(lambda _, r=role: self._pick_color(r))
            self._color_buttons[role] = button
            layout.addRow(label, button)

        self.size_spin = QSpinBox()
        self.size_spin.setRange(*MarkerStyle.MARKER_SIZE_RANGE)
        self.size_spin.setSuffix(' px')
        self.size_spin.setValue(style.marker_size)
        self.size_spin.valueChanged.connect(self._apply_sizes)
        layout.addRow('마커 크기', self.size_spin)

        self.font_spin = QSpinBox()
        self.font_spin.setRange(*MarkerStyle.FONT_SIZE_RANGE)
        self.font_spin.setSuffix(' px')
        self.font_spin.setValue(style.font_size)
        self.font_spin.valueChanged.connect(self._apply_sizes)
        layout.addRow('글자 크기', self.font_spin)

        buttons = QHBoxLayout()
        reset_button = QPushButton('기본값')
        reset_button.clicked.connect(self._reset)
        close_button = QPushButton('닫기')
        close_button.clicked.connect(self.close)
        buttons.addWidget(reset_button)
        buttons.addWidget(close_button)
        layout.addRow(buttons)

        self._sync_from_style()

    def _pick_color(self, role):
        style = marker_style()
        color = QColorDialog.getColor(style.color(role), self, '색상 선택')
        if color.isValid():
            style.set_color(role, color)
            self._sync_from_style()

    def _apply_sizes(self):
        marker_style().set_sizes(self.size_spin.value(), self.font_spin.value())

    def _reset(self):
        marker_style().reset()
        self._sync_from_style()

    def _sync_from_style(self):
        """색 버튼 배경과 스핀박스 값을 현재 설정에 맞춘다."""
        style = marker_style()
        for role, button in self._color_buttons.items():
            color = style.color(role)
            # 배경색이 어두우면 글자는 흰색으로 (버튼에 색상 코드를 함께 보여준다)
            text_color = '#ffffff' if color.lightness() < 128 else '#000000'
            button.setStyleSheet(
                f'QPushButton {{ background-color: {color.name()}; color: {text_color}; }}')
            button.setText(color.name())
        for spin, value in ((self.size_spin, style.marker_size), (self.font_spin, style.font_size)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)


def _qimage_to_rgb(image):
    """QImage를 (h, w, 3) uint8 RGB 배열로 변환한다.

    포맷을 RGB888로 통일하므로 그레이스케일/팔레트/알파 영상도 안전하다.
    스캔라인(stride)은 4바이트 정렬이라 width*3보다 클 수 있어서
    행 단위로 먼저 자른 뒤 (h, w, 3)으로 편다.
    """
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    height, width, stride = image.height(), image.width(), image.bytesPerLine()
    bits = image.constBits()
    bits.setsize(height * stride)
    rows = np.frombuffer(bits, np.uint8).reshape(height, stride)
    return rows[:, :width * 3].reshape(height, width, 3).copy()


def pixmap_to_bgr(pixmap):
    """픽스맵을 OpenCV용 (h, w, 3) uint8 BGR 배열로 변환한다."""
    return np.ascontiguousarray(_qimage_to_rgb(pixmap.toImage())[:, :, ::-1])


# 실시간 정합은 점을 하나 찍을 때마다 원본을 다시 읽으므로,
# 같은 픽스맵의 변환 결과를 몇 장 캐시해 큰 영상에서의 지연을 줄인다.
_BGR_CACHE = {}
_BGR_CACHE_LIMIT = 4


def pixmap_to_bgr_cached(pixmap):
    """pixmap_to_bgr의 캐시 버전. 반환 배열은 수정하지 말 것(공유됨)."""
    key = pixmap.cacheKey()
    cached = _BGR_CACHE.get(key)
    if cached is None:
        if len(_BGR_CACHE) >= _BGR_CACHE_LIMIT:
            _BGR_CACHE.clear()
        cached = pixmap_to_bgr(pixmap)
        _BGR_CACHE[key] = cached
    return cached


def _pixmap_from_bgr(np_img):
    """OpenCV BGR 배열을 픽스맵으로 변환한다."""
    rgb = np.ascontiguousarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    height, width, _ = rgb.shape
    buffer = rgb.tobytes()  # QImage는 복사하지 않으므로 참조를 유지한다
    image = QImage(buffer, width, height, 3 * width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image)


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

        # 마커 색/크기 설정이 바뀌면 이미 찍혀 있는 점들도 함께 갱신한다
        marker_style().changed.connect(self.refresh_markers)

    def init_ui(self):
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.image_item = None
        # 정합점 마커(십자 + 번호). 인덱스가 coordinates와 1:1로 대응한다.
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

        # 정합 결과(오버레이) 창처럼 점을 찍을 필요가 없는 뷰는 읽기 전용으로 둔다
        self.read_only = False

        # 좌클릭 드래그/클릭 구분용 변수
        self._left_press_pos = None
        self._left_moved = False
        self._click_drag_threshold = 4  # 이 픽셀 이상 움직이면 클릭이 아닌 드래그로 간주
        
        # Undo 관련 변수
        self.undo_stack = []

        # 밝기/대비 조정 (화면 표시 전용).
        # _source_pixmap/_source_rgb는 조정 전 원본이며, 저장과 정합은 이쪽을 쓴다.
        self._source_pixmap = None
        self._source_rgb = None
        self._brightness = 0
        self._contrast = 0

    def _mark_dirty(self, dirty=True):
        self._dirty = dirty
        window = self.window()
        if isinstance(window, Image_Window):
            window.update_title()

    def load_image(self, file_name):
        self.scene.clear()
        self.number_items.clear()
        self.coordinates.clear()
        self.number_count = 0
        self._mark_dirty(False)
        # tRNS 청크가 있는 PNG는 특정 색(예: 검정)이 투명으로 로드되어
        # 씬의 흰 배경이 비쳐 보인다. 알파를 제거해 원본 색을 그대로 표시한다.
        image = QImage(file_name)
        if image.hasAlphaChannel():
            image = image.convertToFormat(QImage.Format.Format_RGB32)
        pixmap = QPixmap.fromImage(image)
        self._set_source(pixmap)
        if not pixmap.isNull():
            self.image_item = QGraphicsPixmapItem(self._adjusted_pixmap())
            self.scene.addItem(self.image_item)

            self.setSceneRect(pixmap.rect().x(), pixmap.rect().y(), pixmap.rect().width(), pixmap.rect().height())
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        try:
            update_registration_button()
        except NameError:
            pass

    def load_from_numpy(self, np_img):
        pixmap = _pixmap_from_bgr(np_img)
        self.scene.clear()
        self.number_items.clear()
        self.coordinates.clear()
        self.number_count = 0
        self._set_source(pixmap)
        self.image_item = QGraphicsPixmapItem(self._adjusted_pixmap())
        self.scene.addItem(self.image_item)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._mark_dirty(False)

    def update_from_numpy(self, np_img):
        """줌/스크롤 상태를 유지한 채 표시 이미지만 교체한다 (실시간 갱신용).

        영상 크기가 달라졌을 때만 씬을 새로 만들어 화면에 맞추고 True를
        돌려준다. 이때는 씬에 얹어 둔 마커도 함께 사라지므로 호출한 쪽에서
        다시 만들어야 한다.
        """
        pixmap = _pixmap_from_bgr(np_img)
        if (self.image_item is None or self._source_pixmap is None
                or self._source_pixmap.size() != pixmap.size()):
            self.load_from_numpy(np_img)
            return True
        self._set_source(pixmap)
        self.image_item.setPixmap(self._adjusted_pixmap())
        return False

    # ----- 밝기/대비 조정 (화면 표시 전용) -----

    def _set_source(self, pixmap):
        """조정 전 원본 픽스맵을 보관한다. RGB 배열은 첫 조정 때 만든다."""
        self._source_pixmap = None if pixmap.isNull() else pixmap
        self._source_rgb = None

    def _source_rgb_array(self):
        """원본 픽스맵을 (h, w, 3) uint8 RGB 배열로 변환해 캐시한다."""
        if self._source_rgb is None and self._source_pixmap is not None:
            self._source_rgb = _qimage_to_rgb(self._source_pixmap.toImage())
        return self._source_rgb

    def _adjusted_pixmap(self):
        """현재 밝기/대비를 적용한 픽스맵. 조정값이 0이면 원본을 그대로 쓴다."""
        if self._source_pixmap is None:
            return QPixmap()
        if self._brightness == 0 and self._contrast == 0:
            return self._source_pixmap

        rgb = self._source_rgb_array()
        if rgb is None:
            return self._source_pixmap

        adjusted = np.ascontiguousarray(_build_adjust_lut(self._brightness, self._contrast)[rgb])
        height, width, _ = adjusted.shape
        buffer = adjusted.tobytes()  # QImage가 복사하지 않으므로 참조를 유지한다
        image = QImage(buffer, width, height, 3 * width, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(image)

    def set_adjustment(self, brightness, contrast):
        """밝기/대비를 -100~100 범위로 설정하고 화면을 갱신한다."""
        self._brightness = brightness
        self._contrast = contrast
        if self.image_item is not None and self._source_pixmap is not None:
            self.image_item.setPixmap(self._adjusted_pixmap())

    def source_pixmap(self):
        """조정 전 원본 픽스맵. 정합 입력으로 쓴다."""
        if self._source_pixmap is not None:
            return self._source_pixmap
        return self.image_item.pixmap() if self.image_item is not None else QPixmap()

    @contextlib.contextmanager
    def showing_source_pixmap(self):
        """저장용 렌더링 동안만 원본 픽셀을 표시한다."""
        if self.image_item is None or self._source_pixmap is None:
            yield
            return
        adjusted = self.image_item.pixmap()
        self.image_item.setPixmap(self._source_pixmap)
        try:
            yield
        finally:
            self.image_item.setPixmap(adjusted)

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
        if self.read_only:
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

    def _add_marker(self, x, y, label):
        """(x, y)에 정합점 마커를 놓는다. 마커는 줌과 무관하게 같은 크기로 보인다."""
        item = PointMarkerItem(label=str(label), role='point', shape='cross')
        item.setPos(x, y)
        self.scene.addItem(item)
        self.number_items.append(item)
        return item

    def refresh_markers(self):
        """마커 색/크기 설정이 바뀌었을 때 씬 위의 모든 마커를 다시 그린다."""
        for item in self.scene.items():
            if isinstance(item, PointMarkerItem):
                item.style_changed()

    def Click_Coordinate(self, pos):
        # Undo를 위한 현재 상태 저장
        self.save_state_for_undo()

        self._add_marker(pos.x(), pos.y(), self.number_count + 1)
        self.number_count += 1
        self.coordinates.append(((pos.x()), (pos.y())))
        self._mark_dirty(True)
        try:
            update_registration_button()
        except NameError:
            pass

    # 좌표 전체 삭제
    def remove_cross_items(self):
        for item in self.number_items:
            self.scene.removeItem(item)

        self.coordinates = []
        self.number_items = []
        self.number_count = 0
        self._mark_dirty(True)
        try:
            update_registration_button()
        except NameError:
            pass

    def remove_cross_one_item(self, index):
        if 0 <= index < len(self.number_items):
            self.scene.removeItem(self.number_items[index])
            self.number_items.pop(index)
            self.number_count -= 1
        try:
            update_registration_button()
        except NameError:
            pass

    # 좌표 개별 삭제
    def remove_coordinates(self, pos):
        if self.read_only:
            return False
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
        # 좌표와 레이블만 남기면 마커는 언제든 다시 만들 수 있다
        state = {
            'coordinates': self.coordinates.copy(),
            'number_count': self.number_count,
            'labels': [item.toPlainText() for item in self.number_items],
        }

        self.undo_stack.append(state)
        # Undo 스택 크기 제한 (메모리 절약)
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    # Undo 기능 실행
    def undo(self):
        if not self.undo_stack:
            return

        # 현재 모든 마커 제거
        for item in self.number_items:
            self.scene.removeItem(item)

        # 이전 상태 복원
        state = self.undo_stack.pop()
        self.coordinates = state['coordinates']
        self.number_count = state['number_count']
        self.number_items = []

        for (x, y), label in zip(self.coordinates, state['labels']):
            self._add_marker(x, y, label)
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
        self._add_marker(x, y, index)
        self.number_count += 1
        self.coordinates.append((float(x), float(y)))
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

RANSAC_PARAMS = dict(
    method=cv2.RANSAC,
    ransacReprojThreshold=3.5,
    maxIters=2000,
    confidence=0.995,
)


def _translation_matrix(dx, dy):
    return np.array([[1.0, 0.0, dx],
                     [0.0, 1.0, dy],
                     [0.0, 0.0, 1.0]])


def _similarity_from_two_pairs(points1, points2):
    """두 쌍으로 유사변환(회전+등방 스케일+평행이동)을 구한다.

    두 점을 잇는 벡터를 복소수로 보면 dst/src 나눗셈 한 번에 회전각과
    배율이 함께 나온다. src의 두 점이 겹쳐 방향을 정할 수 없으면 None.
    """
    src_vec = complex(points2[1][0] - points2[0][0], points2[1][1] - points2[0][1])
    dst_vec = complex(points1[1][0] - points1[0][0], points1[1][1] - points1[0][1])
    if abs(src_vec) < 1e-9:
        return None
    z = dst_vec / src_vec
    a, b = z.real, z.imag
    x2, y2 = points2[0]
    tx = points1[0][0] - (a * x2 - b * y2)
    ty = points1[0][1] - (b * x2 + a * y2)
    return np.array([[a, -b, tx],
                     [b, a, ty],
                     [0.0, 0.0, 1.0]])


def _is_usable_transform(matrix):
    """유한한 값이고 면적이 0으로 붕괴하지 않는(퇴화하지 않은) 변환인지 확인."""
    if matrix is None or not np.all(np.isfinite(matrix)):
        return False
    return abs(float(np.linalg.det(np.asarray(matrix)[:2, :2]))) > 1e-9


def estimate_live_transform(points1, points2, size1, size2):
    """정합점이 늘어나는 만큼 자유도를 올려가며 img2 → img1 변환을 추정한다.

    정합점이 부족한 단계에서 호모그래피(8자유도)를 억지로 풀면 해가
    발산하므로, 쌍 개수에 맞는 최소 모델을 쓴다.

      0쌍  : 두 영상의 중앙이 겹치도록 평행이동 (1:1 오버레이)
      1쌍  : 그 점이 겹치도록 평행이동
      2쌍  : 유사변환(회전 + 등방 스케일 + 평행이동)
      3쌍  : 어파인
      4쌍  : 최소제곱 호모그래피
      5쌍~ : RANSAC 호모그래피 ('정합' 버튼과 동일한 계산)

    반환: (3x3 변환행렬, inlier 마스크 또는 None, 사용한 모델 이름)
    """
    count = len(points1)
    if count == 0:
        (width1, height1), (width2, height2) = size1, size2
        return _translation_matrix((width1 - width2) / 2.0, (height1 - height2) / 2.0), None, '중앙 정렬'

    p1 = np.asarray(points1, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(points2, dtype=np.float64).reshape(-1, 2)

    if count >= 2:
        src = p2.astype(np.float32).reshape(-1, 1, 2)
        dst = p1.astype(np.float32).reshape(-1, 1, 2)

        # 퇴화한 배치(한 줄 위의 점, 겹친 점 등)에서는 OpenCV가 예외를 던지거나
        # 쓸 수 없는 행렬을 주므로, 실패하면 아래에서 낮은 모델로 물러난다.
        try:
            if count == 2:
                matrix = _similarity_from_two_pairs(p1, p2)
                if _is_usable_transform(matrix):
                    return matrix, None, '유사변환'
            elif count == 3:
                affine = cv2.getAffineTransform(src.reshape(3, 2), dst.reshape(3, 2))
                if _is_usable_transform(affine):
                    return np.vstack([affine, [0.0, 0.0, 1.0]]), None, '어파인'
            elif count == 4:
                matrix, _ = cv2.findHomography(src, dst, 0)
                if _is_usable_transform(matrix):
                    return matrix, None, '호모그래피'
            else:
                matrix, inliers = cv2.findHomography(src, dst, **RANSAC_PARAMS)
                if _is_usable_transform(matrix):
                    return matrix, inliers, '호모그래피(RANSAC)'
        except cv2.error:
            pass

        matrix = _similarity_from_two_pairs(p1, p2)
        if _is_usable_transform(matrix):
            return matrix, None, '유사변환(퇴화 보정)'

    offset = p1[0] - p2[0]
    return _translation_matrix(offset[0], offset[1]), None, '평행이동'


def warp_to_reference(moving_img, matrix, size):
    """정합 대상 영상을 주어진 크기의 캔버스로 워핑한다 (바깥은 검정).

    size는 (너비, 높이).
    """
    return cv2.warpPerspective(
        moving_img,
        matrix,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


# 퇴화한 변환 때문에 캔버스가 터무니없이 커지는 것을 막는 상한
CANVAS_AREA_LIMIT = 6.0      # 두 영상 중 넓은 쪽 넓이의 몇 배까지 허용할지
CANVAS_SIDE_LIMIT = 16384    # 한 변의 최대 픽셀 수


def overlay_canvas(base_shape, moving_shape, matrix):
    """기준 영상과 워핑된 영상을 모두 담는 캔버스를 계산한다.

    기준 영상 크기에 맞춰 자르면 그보다 큰 영상이 잘려 나가므로, 두 영상을
    모두 감싸는 사각형을 캔버스로 삼고 그 왼쪽 위가 (0,0)이 되도록 옮긴다.

    반환: (원점 보정 행렬, (너비, 높이)). 변환이 퇴화해 캔버스를 감당할 수
    없을 만큼 키우면 None을 돌려주고, 호출한 쪽이 기준 영상 크기로 되돌아간다.
    """
    height1, width1 = base_shape[:2]
    height2, width2 = moving_shape[:2]

    # perspectiveTransform은 NaN 행렬을 받으면 좌표를 0으로 돌려주므로
    # 결과만 봐서는 이상을 알 수 없다. 행렬 자체를 먼저 확인한다.
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        return None

    corners = np.array([[0.0, 0.0], [width2, 0.0], [width2, height2], [0.0, height2]],
                       dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    if not np.all(np.isfinite(warped)):
        return None

    # 기준 영상의 네 모서리(0,0)-(w1,h1)도 함께 담아야 한다
    xs = np.concatenate([warped[:, 0], [0.0, float(width1)]])
    ys = np.concatenate([warped[:, 1], [0.0, float(height1)]])
    left, top = float(np.floor(xs.min())), float(np.floor(ys.min()))
    width = int(np.ceil(xs.max()) - left)
    height = int(np.ceil(ys.max()) - top)

    area_limit = CANVAS_AREA_LIMIT * max(width1 * height1, width2 * height2)
    if (width <= 0 or height <= 0
            or width > CANVAS_SIDE_LIMIT or height > CANVAS_SIDE_LIMIT
            or width * height > area_limit):
        return None
    return _translation_matrix(-left, -top), (width, height)


def build_overlay_layers(base_img, moving_img, matrix):
    """두 영상을 같은 캔버스 위에 올린다.

    반환: (기준 레이어, 정합된 레이어, 원점 보정 행렬).
    보정 행렬은 기준 영상 좌표를 캔버스 좌표로 옮기는 평행이동이라,
    정합점 위치를 표시할 때도 같이 적용해야 한다.
    """
    canvas = overlay_canvas(base_img.shape, moving_img.shape, matrix)
    if canvas is None:
        # 퇴화한 변환이면 예전처럼 기준 영상 크기로 자른다
        offset = np.eye(3)
        size = (base_img.shape[1], base_img.shape[0])
    else:
        offset, size = canvas

    width, height = size
    base_layer = np.zeros((height, width, 3), dtype=np.uint8)
    x0, y0 = int(round(offset[0, 2])), int(round(offset[1, 2]))
    # 기준 영상은 평행이동만 하므로 워핑 대신 슬라이싱으로 붙인다(정확하고 빠르다).
    # 캔버스는 기준 영상을 포함하도록 잡혔지만, 퇴화 대비로 잘라 넣는다.
    patch = base_img[:height - y0, :width - x0]
    base_layer[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]] = patch

    warped_layer = warp_to_reference(moving_img, offset @ np.asarray(matrix, dtype=np.float64), size)
    return base_layer, warped_layer, offset


def transform_points(points, matrix):
    """정합점들을 변환행렬로 옮긴다."""
    if len(points) == 0:
        return np.empty((0, 2))
    reshaped = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(reshaped, matrix.astype(np.float64)).reshape(-1, 2)


def registration_status(keys, points1, registered_points2, inliers, model_name):
    """상태 표시줄 문구: 쌍 개수, 사용한 모델, inlier 수, RMSE.

    RANSAC이 걸러낸 outlier는 잔차가 수백 픽셀까지 나올 수 있어서, 전체
    평균을 쓰면 정작 정합이 잘 됐는지 알 수 없다. inlier 기준으로 RMSE를
    내고 걸러진 개수를 함께 보여 준다(outlier는 화면에서 빨간 마커).
    """
    parts = [f'정합점 {len(keys)}쌍', model_name]
    if len(keys):
        residuals = np.linalg.norm(np.asarray(registered_points2) - np.asarray(points1), axis=1)
        if inliers is not None:
            mask = np.asarray(inliers).ravel().astype(bool)
            parts.append(f'inlier {int(mask.sum())}/{len(keys)}')
            if mask.any():
                residuals = residuals[mask]
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        parts.append(f'RMSE {rmse:.2f} px')
    return ' · '.join(parts)


def _resolve_sibling_dir(parent_dir, folder_name):
    """parent_dir 아래에서 folder_name에 해당하는 폴더의 실제 경로를 반환.

    리눅스는 폴더명 대소문자를 구분하므로 'EO'/'eo' 같은 표기 차이를 흡수한다.
    """
    exact = f"{parent_dir}/{folder_name}"
    if os.path.isdir(exact):
        return exact
    try:
        entries = os.listdir(parent_dir)
    except OSError:
        return None
    for entry in entries:
        if entry.lower() == folder_name.lower():
            candidate = f"{parent_dir}/{entry}"
            if os.path.isdir(candidate):
                return candidate
    return None


def _find_tile_counterpart(normalized):
    """neonsat_L1G ↔ google_ref: 타일 식별자(R###_C###)로 짝을 찾는다.

    예) .../neonsat_google_tie_points/neonsat_L1G/neonsat_L1G_R001_C003.png
      ↔ .../neonsat_google_tie_points/google_ref/google_ref_R001_C003.tif

    두 폴더는 같은 부모(neonsat_google_tie_points) 아래의 형제 폴더이며,
    폴더마다 접두사와 확장자가 다르므로(neonsat은 .png, google은 .tif)
    식별자만으로 매칭한다. 파일명이 '{폴더명}_{타일ID}' 규칙을 따르므로
    대응 파일명을 그대로 조립할 수 있다.
    """
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
            return counterpart

    return None


# _REG_###### 인덱스로 짝을 찾는 형제 폴더 쌍.
# 이 폴더들은 파일명 접두사가 서로 달라(SAR은 K5_..., EO는 K3A_...)
# 대응 파일명을 조립할 수 없고, 인덱스로 탐색해야 한다.
REG_INDEX_FOLDER_PAIRS = (('SAR', 'EO'),)


def _find_reg_index_counterpart(normalized):
    """SAR ↔ EO: 같은 부모 아래 형제 폴더에서 _REG_###### 인덱스로 짝을 찾는다.

    예) .../K5_20190522094059/SAR/K5_20190522094059_REG_000001.png
      ↔ .../K5_20190522094059/EO/K3A_20200320050332_REG_000001.png

    인덱스는 자릿수 표기 차이(_REG_000001 vs _REG_1)를 흡수하도록 정수로
    비교한다. 후보가 둘 이상이면 어느 쪽이 짝인지 특정할 수 없으므로
    자동으로 열지 않는다.
    """
    basename = os.path.basename(normalized)

    match = re.search(r'_REG_(\d+)', basename, re.IGNORECASE)
    if not match:
        return None
    index = int(match.group(1))

    src_dir = os.path.dirname(normalized)
    parent_dir = os.path.dirname(src_dir)
    src_folder = os.path.basename(src_dir)

    # 현재 폴더가 쌍의 어느 쪽인지 판별하고 반대쪽 폴더명을 얻는다
    target_folder = None
    for first, second in REG_INDEX_FOLDER_PAIRS:
        if src_folder.lower() == first.lower():
            target_folder = second
            break
        if src_folder.lower() == second.lower():
            target_folder = first
            break
    if target_folder is None:
        return None

    target_dir = _resolve_sibling_dir(parent_dir, target_folder)
    if target_dir is None:
        return None

    try:
        entries = sorted(os.listdir(target_dir))
    except OSError:
        return None

    candidates = []
    for entry in entries:
        if not entry.lower().endswith(ImageViewer.IMAGE_EXTENSIONS):
            continue
        entry_match = re.search(r'_REG_(\d+)', entry, re.IGNORECASE)
        if entry_match and int(entry_match.group(1)) == index:
            candidates.append(f"{target_dir}/{entry}")

    # 짝이 유일할 때만 연다
    if len(candidates) != 1:
        return None
    return candidates[0]


def find_counterpart_image(file_path):
    """열려 있는 이미지의 대응 이미지 경로를 반환. 못 찾으면 None.

    두 가지 폴더 규칙을 순서대로 시도한다.
      1) neonsat_L1G ↔ google_ref : 타일 식별자(R###_C###)로 매칭
      2) SAR ↔ EO                 : _REG_###### 인덱스로 매칭
    """
    normalized = file_path.replace('\\', '/')

    for finder in (_find_tile_counterpart, _find_reg_index_counterpart):
        counterpart = finder(normalized)
        if counterpart:
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
    # 종료가 확정되면 세워지는 표식. Qt가 남은 창들을 닫을 때 종료 확인을
    # 다시 묻지 않도록 모든 창이 함께 본다.
    _quitting = False

    def __init__(self, settings_key=None):
        super().__init__()
        # settings_key가 있는 창만 밝기/대비를 저장·복원한다.
        # 오버레이 결과 창처럼 일시적인 창은 None으로 두어 값을 남기지 않는다.
        self._settings_key = settings_key
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

        self.add_adjust_controls(toolbar)

        self.statusBar()

    def add_adjust_controls(self, toolbar):
        """툴바 오른쪽 끝에 밝기/대비 슬라이더를 붙인다 (창마다 독립)."""
        # 신축 여백을 넣어 이후 위젯을 오른쪽으로 민다
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self.brightness_slider, self.brightness_value = self._add_adjust_slider(toolbar, 'Brightness')
        self.contrast_slider, self.contrast_value = self._add_adjust_slider(toolbar, 'Contrast')

        reset_button = QPushButton('Reset')
        reset_button.setToolTip('밝기/대비를 원본으로 되돌립니다')
        reset_button.clicked.connect(self.reset_adjustment)
        toolbar.addWidget(reset_button)

        self.add_marker_style_button(toolbar)
        self._restore_adjustment()

    # 마커 설정 창에 띄울 색상 항목 (창 종류마다 다르다)
    MARKER_STYLE_ROLES = (('point', '정합점 색상'),)

    def add_marker_style_button(self, toolbar):
        """정합점 마커의 색/크기 설정 창을 여는 버튼."""
        button = QPushButton('Marker')
        button.setToolTip('정합점 마커와 번호의 색상·크기를 설정합니다 (줌과 무관하게 같은 크기로 보입니다)')
        button.clicked.connect(self.open_marker_style_dialog)
        toolbar.addWidget(button)

    def open_marker_style_dialog(self):
        # 모덜리스로 띄워 설정을 바꾸면서 결과를 바로 확인할 수 있게 한다
        if getattr(self, '_marker_dialog', None) is None:
            self._marker_dialog = MarkerStyleDialog(self, self.MARKER_STYLE_ROLES)
        self._marker_dialog.show()
        self._marker_dialog.raise_()
        self._marker_dialog.activateWindow()

    def _add_adjust_slider(self, toolbar, label_text):
        """라벨 + 슬라이더 + 현재값 라벨 한 벌을 툴바에 추가한다."""
        label = QLabel(f' {label_text} ')
        toolbar.addWidget(label)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(-100, 100)
        slider.setValue(0)
        slider.setFixedWidth(120)
        slider.setToolTip(f'{label_text} (-100 ~ 100). 화면 표시에만 적용됩니다')
        slider.valueChanged.connect(self._on_adjust_changed)
        toolbar.addWidget(slider)

        # 값 표시 폭을 고정해 슬라이더가 움직일 때 툴바가 흔들리지 않게 한다
        value_label = QLabel('0')
        value_label.setFixedWidth(32)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        toolbar.addWidget(value_label)

        return slider, value_label

    def _on_adjust_changed(self):
        brightness = self.brightness_slider.value()
        contrast = self.contrast_slider.value()
        self.brightness_value.setText(str(brightness))
        self.contrast_value.setText(str(contrast))
        self.viewer.set_adjustment(brightness, contrast)
        self._save_adjustment(brightness, contrast)

    def reset_adjustment(self):
        self._set_sliders(0, 0)

    def _set_sliders(self, brightness, contrast):
        """슬라이더를 조용히 옮긴 뒤 한 번만 반영한다 (중간값 저장 방지)."""
        for slider, value in ((self.brightness_slider, brightness),
                              (self.contrast_slider, contrast)):
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
        self._on_adjust_changed()

    def _save_adjustment(self, brightness, contrast):
        if not self._settings_key:
            return
        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        settings.setValue(f'{self._settings_key}/brightness', brightness)
        settings.setValue(f'{self._settings_key}/contrast', contrast)

    def _restore_adjustment(self):
        """지난 실행에서 쓰던 밝기/대비를 복원한다. 없으면 0으로 둔다."""
        if not self._settings_key:
            return
        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        try:
            brightness = int(settings.value(f'{self._settings_key}/brightness', 0))
            contrast = int(settings.value(f'{self._settings_key}/contrast', 0))
        except (TypeError, ValueError):
            # 설정 파일이 손상돼도 실행은 막지 않는다
            brightness = contrast = 0
        # 슬라이더 범위를 벗어난 값은 setValue가 알아서 잘라낸다
        self._set_sliders(brightness, contrast)

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
            # 밝기/대비는 보기 보조일 뿐이므로 저장은 조정 전 원본 픽셀로 한다.
            with self.viewer.showing_source_pixmap():
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
    
    def _session_windows(self):
        """한 세션을 이루는 원본 창들(자기 자신과 파트너 창)."""
        windows = [self]
        if self.partner_window is not None and self.partner_window is not self:
            windows.append(self.partner_window)
        return windows

    def request_exit(self):
        """종료해도 되는지 한 번만 확인한다.

        Exit 버튼과 창 닫기(X)가 이 한 곳을 거치므로 확인 창이 두 번 뜨지
        않는다. 확인이 끝나면 _quitting을 세워, Qt가 나머지 창을 닫을 때
        다시 묻지 않도록 한다. 저장은 두 창을 모두 살펴 한꺼번에 처리한다.
        """
        if Image_Window._quitting:
            return True

        dirty_windows = [w for w in self._session_windows() if w.viewer._dirty]
        if dirty_windows:
            names = ', '.join(w.folder_name or w.windowTitle() for w in dirty_windows)
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle('종료 확인')
            msg.setText(f'저장되지 않은 좌표가 있습니다.\n[{names}]\n저장하시겠습니까?')
            msg.setStandardButtons(
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel
            )
            msg.setDefaultButton(QMessageBox.StandardButton.Save)
            reply = msg.exec()
            if reply == QMessageBox.StandardButton.Save:
                for window in dirty_windows:
                    if not window.quick_save_coordinates():
                        # 저장이 막히면 종료하지 않고 무결성 문제 해결을 유도
                        return False
            elif reply != QMessageBox.StandardButton.Discard:
                return False
        else:
            reply = QMessageBox.question(self, '종료 확인',
                                       '프로그램을 종료하시겠습니까?',
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                       QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return False

        Image_Window._quitting = True
        return True

    def confirm_exit_application(self):
        if self.request_exit():
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
        # 이미 종료가 확정된 뒤라면 원본 창도 묻지 않고 닫는다.
        if self.is_overlay or Image_Window._quitting:
            event.accept()
            return

        # 원본 창 하나를 닫는 것은 곧 작업 종료다(두 창이 한 벌로 동작한다).
        # Exit 버튼과 같은 경로를 써서 확인 창이 한 번만 뜨게 한다.
        if self.request_exit():
            event.accept()
            QApplication.instance().quit()
        else:
            event.ignore()

class RegistrationOverlayWindow(Image_Window):
    """정합 결과를 겹쳐 보여주는 창.

    기준 영상과 정합된 영상을 각각 들고 있다가 매번 합성하므로,
    불투명도를 바꾸거나 플리커링을 해도 다시 정합할 필요가 없다.
    정합이 잘 맞았는지 보는 것이 목적이라 밝기/대비 대신
    영상별 불투명도와 플리커링(A/B 교대 표시)을 제공한다.
    """

    # 창이 닫힐 때 자기 자신을 넘겨 호출하는 콜백 (생성한 쪽에서 지정)
    on_closed = None
    # 'Save Overlay'의 기본 파일명
    default_save_name = 'overlay.png'

    def __init__(self, live=False):
        super().__init__()
        self.is_overlay = True
        self.viewer.read_only = True   # 결과 확인용 창이므로 점을 찍지 않는다
        self._live = live
        self.setWindowTitle('Live Registration Overlay' if live else 'Overlay Registration Result')
        # 원본 창 두 개를 완전히 가리지 않도록 살짝 비켜서 띄운다
        self.move(200, 100)

    # ----- 툴바 -----

    # 오버레이 창은 inlier/outlier/글자 색을 따로 고를 수 있다
    MARKER_STYLE_ROLES = (
        ('inlier', 'Inlier 색상'),
        ('outlier', 'Outlier 색상'),
        ('text', '번호/잔차 색상'),
    )

    def create_toolbar(self):
        # Image_Window.__init__이 initUI()를 거쳐 여기까지 오므로,
        # 합성에 필요한 상태는 이 시점에 초기화한다.
        self._base_img = None       # 기준 영상 (BGR)
        self._warped_img = None     # 기준 좌표계로 정합된 영상 (BGR)
        self._markers = None        # (points1, registered_points2, inliers, keys)
        self._marker_items = []     # 씬에 얹은 정합점 마커
        self._flicker_mode = None   # None(합성) / 'base' / 'warped'
        # 캔버스 원점(기준 영상이 캔버스 안에서 밀려난 양). 정합점이 늘면서
        # 캔버스 크기가 바뀌어도 보고 있던 자리를 유지하는 데 쓴다.
        self._origin = (0.0, 0.0)
        self._shown_origin = None

        self._flicker_timer = QTimer(self)
        self._flicker_timer.timeout.connect(self._advance_flicker)

        toolbar = QToolBar('toolbar')
        self.addToolBar(toolbar)

        plus_action = QAction(QIcon('./icon/zoom_in.png'), 'Zoom In', self)
        plus_action.triggered.connect(self.zoom_in)
        toolbar.addAction(plus_action)

        minus_action = QAction(QIcon('./icon/zoom_out.png'), 'Zoom Out', self)
        minus_action.triggered.connect(self.zoom_out)
        toolbar.addAction(minus_action)

        save_action = QAction(QIcon('./icon/save.png'), 'Save Overlay', self)
        save_action.setStatusTip('현재 보이는 오버레이 영상을 저장합니다')
        save_action.triggered.connect(self.save_overlay)
        toolbar.addAction(save_action)

        self.add_adjust_controls(toolbar)
        self.statusBar()

    def add_adjust_controls(self, toolbar):
        """밝기/대비 대신 불투명도 슬라이더와 플리커링 조작부를 붙인다."""
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self.opacity1_slider, self.opacity1_value = self._add_opacity_slider(
            toolbar, '영상1', '기준 영상(창 1)의 불투명도')
        self.opacity2_slider, self.opacity2_value = self._add_opacity_slider(
            toolbar, '영상2', '정합된 영상(창 2)의 불투명도')

        # 정합점 표시는 기본으로 켜 두고, 영상만 보고 싶을 때 끌 수 있게 한다
        self.points_button = QPushButton('Points')
        self.points_button.setCheckable(True)
        self.points_button.setChecked(True)
        self.points_button.setToolTip(
            '정합점 마커(번호와 잔차)를 표시합니다.\n'
            '초록 = inlier, 빨강 = outlier, 노랑 숫자 = 잔차(px)')
        self.points_button.setStyleSheet(
            'QPushButton:checked { background-color: #4caf50; color: white; }')
        self.points_button.toggled.connect(self._on_points_toggled)
        toolbar.addWidget(self.points_button)

        self.add_marker_style_button(toolbar)

        self.flicker_button = QPushButton('Flicker')
        self.flicker_button.setCheckable(True)
        self.flicker_button.setToolTip(
            '두 영상을 정해진 간격으로 번갈아 보여줍니다.\n'
            'Space 키로 합성 → 영상1 → 영상2 순으로 직접 넘길 수도 있습니다.')
        self.flicker_button.setStyleSheet(
            'QPushButton:checked { background-color: #4caf50; color: white; }')
        self.flicker_button.toggled.connect(self._on_flicker_toggled)
        toolbar.addWidget(self.flicker_button)

        self.flicker_interval = QDoubleSpinBox()
        self.flicker_interval.setRange(0.1, 10.0)
        self.flicker_interval.setSingleStep(0.1)
        self.flicker_interval.setDecimals(1)
        self.flicker_interval.setValue(0.5)
        self.flicker_interval.setSuffix(' s')
        self.flicker_interval.setFixedWidth(70)
        self.flicker_interval.setToolTip('플리커링 간격(초)')
        self.flicker_interval.valueChanged.connect(self._on_interval_changed)
        toolbar.addWidget(self.flicker_interval)

    def _add_opacity_slider(self, toolbar, label_text, tooltip):
        label = QLabel(f' {label_text} ')
        toolbar.addWidget(label)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(50)
        slider.setFixedWidth(120)
        slider.setToolTip(f'{tooltip} (0 ~ 100%)')
        slider.valueChanged.connect(self._on_opacity_changed)
        toolbar.addWidget(slider)

        # 값 표시 폭을 고정해 슬라이더가 움직일 때 툴바가 흔들리지 않게 한다
        value_label = QLabel('50')
        value_label.setFixedWidth(32)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        toolbar.addWidget(value_label)

        return slider, value_label

    # ----- 레이어 합성 -----

    def set_layers(self, base_img, warped_img, markers=None, status='', origin=(0.0, 0.0)):
        """기준/정합 레이어를 교체하고 화면을 다시 그린다.

        origin은 기준 영상이 캔버스 안에서 밀려난 양이다. 정합점이 늘면서
        캔버스가 커지면 같은 지형이 다른 씬 좌표로 옮겨가는데, 이 값으로
        보정해 사용자가 보고 있던 자리를 그대로 지킨다.
        base_img가 None이면 아직 보여줄 것이 없다는 뜻이라 상태만 알린다.
        """
        self._base_img = base_img
        self._warped_img = warped_img
        self._markers = markers
        self._origin = (float(origin[0]), float(origin[1]))
        if status:
            self.statusBar().showMessage(status)
        if base_img is None or warped_img is None:
            # 보여줄 것이 없으면 남아 있던 마커도 함께 지운다
            self._rebuild_markers()
            return
        self._refresh_view()

    def _compose(self):
        """현재 불투명도/플리커링 상태에 맞는 합성 영상을 만든다."""
        if self._base_img is None or self._warped_img is None:
            return None

        if self._flicker_mode == 'base':
            return self._base_img
        if self._flicker_mode == 'warped':
            return self._warped_img

        # 검정 배경 위에 두 영상을 각자의 불투명도로 얹는다.
        # 워핑 바깥은 0이므로 그 영역은 기준 영상만 남는다.
        alpha1 = self.opacity1_slider.value() / 100.0
        alpha2 = self.opacity2_slider.value() / 100.0
        return cv2.addWeighted(self._base_img, alpha1, self._warped_img, alpha2, 0)

    def _refresh_view(self):
        composed = self._compose()
        if composed is None:
            return

        viewer = self.viewer
        had_image = viewer.image_item is not None
        transform = viewer.transform()
        center = viewer.mapToScene(viewer.viewport().rect().center())
        shift_x = self._origin[0] - self._shown_origin[0] if self._shown_origin else 0.0
        shift_y = self._origin[1] - self._shown_origin[1] if self._shown_origin else 0.0

        # 씬이 새로 만들어졌으면 얹어 두었던 마커도 함께 사라진 상태다
        rebuilt = viewer.update_from_numpy(composed)
        if rebuilt:
            self._marker_items = []

        # 캔버스 크기가 바뀌면 fitInView로 화면이 초기화되므로, 첫 표시가
        # 아니라면 보고 있던 배율과 위치를 되돌려 놓는다.
        if had_image and (rebuilt or shift_x or shift_y):
            viewer.setTransform(transform)
            viewer.centerOn(center.x() + shift_x, center.y() + shift_y)

        self._shown_origin = self._origin
        self._rebuild_markers()

    def _rebuild_markers(self):
        """정합점 마커를 씬 위에 다시 얹는다.

        영상에 그려 넣지 않고 씬 아이템으로 두기 때문에, 축소해도 마커와
        번호가 같이 작아지지 않고 불투명도/플리커링과도 섞이지 않는다.
        """
        for item in self._marker_items:
            self.viewer.scene.removeItem(item)
        self._marker_items = []

        if not self._markers or not self.points_button.isChecked():
            return

        points1, points2, inliers, keys = self._markers
        for i, ((x1, y1), (x2, y2)) in enumerate(zip(points1, points2)):
            is_inlier = True if inliers is None else bool(inliers[i])
            label = str(keys[i]) if keys is not None else str(i + 1)
            # 잔차(기준점과 얼마나 어긋났는지)를 함께 보여 준다
            residual = float(np.hypot(x1 - x2, y1 - y2))
            item = PointMarkerItem(
                label=label, sub_label=f'{residual:.1f}',
                role='inlier' if is_inlier else 'outlier',
                text_role='text', shape='circle')
            item.setPos(float(x2), float(y2))
            self.viewer.scene.addItem(item)
            self._marker_items.append(item)

    def _on_points_toggled(self, checked):
        self._rebuild_markers()

    def _on_opacity_changed(self):
        self.opacity1_value.setText(str(self.opacity1_slider.value()))
        self.opacity2_value.setText(str(self.opacity2_slider.value()))
        self._refresh_view()

    # ----- 플리커링 -----

    def _on_flicker_toggled(self, checked):
        if checked:
            self._flicker_mode = 'base'
            self._flicker_timer.start(int(self.flicker_interval.value() * 1000))
        else:
            self._flicker_timer.stop()
            self._flicker_mode = None
        self._update_opacity_enabled()
        self._refresh_view()

    def _advance_flicker(self):
        self._flicker_mode = 'warped' if self._flicker_mode == 'base' else 'base'
        self._refresh_view()

    def _on_interval_changed(self, value):
        if self._flicker_timer.isActive():
            self._flicker_timer.start(int(value * 1000))

    def _update_opacity_enabled(self):
        """한쪽만 보여주는 동안에는 불투명도 슬라이더가 의미 없으므로 비활성화."""
        enabled = self._flicker_mode is None
        for widget in (self.opacity1_slider, self.opacity2_slider,
                       self.opacity1_value, self.opacity2_value):
            widget.setEnabled(enabled)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            # 자동 플리커링 중이면 멈추고, 사용자가 직접 한 단계씩 넘긴다
            if self.flicker_button.isChecked():
                self.flicker_button.setChecked(False)
            order = (None, 'base', 'warped')
            self._flicker_mode = order[(order.index(self._flicker_mode) + 1) % len(order)]
            self._update_opacity_enabled()
            self._refresh_view()
            return
        super().keyPressEvent(event)

    # ----- 저장/종료 -----

    def save_overlay(self):
        if self._base_img is None:
            return
        dialog = QFileDialog(self, 'Save Overlay Image', self.default_save_name)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilter('Images (*.png *.jpg *.bmp);;All Files (*)')
        dialog.setDefaultSuffix('png')
        dialog.setOptions(QFileDialog.Option.DontUseNativeDialog)
        if dialog.exec():
            file_name = dialog.selectedFiles()[0]
            # 화면에 보이는 그대로(현재 불투명도/플리커 상태 + 정합점 마커) 저장한다.
            # 마커는 씬 아이템이므로 씬을 1:1로 렌더링해야 함께 담긴다.
            scene_rect = self.viewer.sceneRect()
            pixmap = QPixmap(scene_rect.size().toSize())
            pixmap.fill(Qt.GlobalColor.black)
            painter = QPainter(pixmap)
            self.viewer.scene.render(painter, QRectF(pixmap.rect()), scene_rect)
            painter.end()
            pixmap.save(file_name)

    def closeEvent(self, event):
        self._flicker_timer.stop()
        if self.on_closed is not None:
            self.on_closed(self)
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    Window_one = Image_Window(settings_key='window1')
    Window_one.show()

    Window_two = Image_Window(settings_key='window2')
    Window_two.setWindowTitle("Image Registration Tool 2")
    Window_two.move(350, 0)

    Window_one.partner_window = Window_two
    Window_two.partner_window = Window_one
    Window_two.show()

    active_overlays = []
    # Live 오버레이 창과, 불필요한 재계산을 막기 위한 마지막 입력 지문
    live_state = {'window': None, 'signature': None}

    def common_point_pairs():
        """두 창에서 같은 인덱스로 짝지어진 정합점을 인덱스 순으로 모은다.

        반환: (인덱스 리스트, 창1 좌표 (n,2), 창2 좌표 (n,2))
        """
        try:
            points1_dict = {int(n.toPlainText()): [c[0], c[1]] for n, c in zip(Window_one.viewer.number_items, Window_one.viewer.coordinates)}
            points2_dict = {int(n.toPlainText()): [c[0], c[1]] for n, c in zip(Window_two.viewer.number_items, Window_two.viewer.coordinates)}
        except ValueError:
            # 정수가 아닌 레이블이 있으면 짝을 지을 수 없다
            return [], np.empty((0, 2)), np.empty((0, 2))

        keys = sorted(set(points1_dict) & set(points2_dict))
        points1 = np.array([points1_dict[k] for k in keys], dtype=np.float64).reshape(-1, 2)
        points2 = np.array([points2_dict[k] for k in keys], dtype=np.float64).reshape(-1, 2)
        return keys, points1, points2

    def get_common_count():
        return len(common_point_pairs()[0])

    def refresh_live_overlay(force=False):
        """정합점/영상이 바뀔 때마다 Live 오버레이를 다시 계산한다.

        완성된 정합점 쌍(양쪽에 같은 인덱스가 있는 점)만 사용하므로,
        한쪽에만 점을 찍은 동안에는 화면이 바뀌지 않고, 짝의 한쪽을 지우면
        그 쌍이 빠진 직전 상태로 자연스럽게 되돌아간다.
        """
        window = live_state['window']
        if window is None:
            return

        if Window_one.viewer.image_item is None or Window_two.viewer.image_item is None:
            live_state['signature'] = None
            window.set_layers(None, None, status='두 창 모두 영상을 열어야 실시간 정합이 됩니다.')
            return

        base_pixmap = Window_one.viewer.source_pixmap()
        moving_pixmap = Window_two.viewer.source_pixmap()
        keys, points1, points2 = common_point_pairs()

        # 입력이 그대로면(예: 아직 짝이 안 맞는 점만 찍은 경우) 다시 워핑하지 않는다
        signature = (base_pixmap.cacheKey(), moving_pixmap.cacheKey(),
                     tuple(keys), points1.tobytes(), points2.tobytes())
        if not force and signature == live_state['signature']:
            return
        live_state['signature'] = signature

        # 밝기/대비는 보기 보조일 뿐이므로 정합에는 조정 전 원본을 넣는다
        base_img = pixmap_to_bgr_cached(base_pixmap)
        moving_img = pixmap_to_bgr_cached(moving_pixmap)

        matrix, inliers, model_name = estimate_live_transform(
            points1, points2,
            (base_img.shape[1], base_img.shape[0]),
            (moving_img.shape[1], moving_img.shape[0]))
        base_layer, warped_layer, offset = build_overlay_layers(base_img, moving_img, matrix)

        markers = None
        registered_points2 = None
        canvas_points1 = None
        inlier_mask = None if inliers is None else inliers.ravel()
        if len(keys):
            # 정합점도 캔버스 좌표로 옮겨서 찍는다
            canvas_points1 = points1 + [offset[0, 2], offset[1, 2]]
            registered_points2 = transform_points(points2, offset @ matrix)
            markers = (canvas_points1, registered_points2, inlier_mask, keys)

        status = registration_status(keys, canvas_points1, registered_points2,
                                     inlier_mask, model_name)
        window.set_layers(base_layer, warped_layer, markers, status,
                          origin=(offset[0, 2], offset[1, 2]))

    def update_registration_button():
        reg_action.setEnabled(get_common_count() >= 5)
        try:
            refresh_live_overlay()
        except Exception as exc:
            # 실시간 미리보기 실패가 점 찍기 자체를 막지 않도록 한다
            Window_two.statusBar().showMessage(f'실시간 정합 실패: {exc}', 5000)

    # Add registration button to Window_two's toolbar
    toolbar = Window_two.findChild(QToolBar)
    reg_action = QAction(QIcon('./icon/reg_img.png'), "Perform Registration", Window_two)
    reg_action.setStatusTip("Register images and show overlay")
    toolbar.addAction(reg_action)

    # 정합 버튼 옆의 Live 토글: 켜면 정합점이 바뀔 때마다 오버레이가 즉시 갱신된다
    live_button = QPushButton('Live')
    live_button.setCheckable(True)
    live_button.setToolTip(
        '실시간 정합 미리보기를 켭니다.\n'
        '정합점이 없으면 두 영상의 중앙을 맞춰 1:1로 겹쳐 보여주고,\n'
        '정합점 쌍이 하나씩 완성될 때마다 변환을 다시 계산해 반영합니다.')
    live_button.setStyleSheet('QPushButton:checked { background-color: #4caf50; color: white; }')
    toolbar.addWidget(live_button)

    def on_live_overlay_closed(window):
        live_state['window'] = None
        live_state['signature'] = None
        live_button.blockSignals(True)
        live_button.setChecked(False)
        live_button.blockSignals(False)

    def toggle_live(checked):
        if checked:
            if live_state['window'] is None:
                window = RegistrationOverlayWindow(live=True)
                window.default_save_name = (
                    f"{Window_two.folder_name}_live_overlay.png" if Window_two.folder_name else 'overlay.png')
                window.on_closed = on_live_overlay_closed
                live_state['window'] = window
                window.show()
            refresh_live_overlay(force=True)
        elif live_state['window'] is not None:
            # closeEvent에서 on_live_overlay_closed가 상태를 정리한다
            live_state['window'].close()

    live_button.toggled.connect(toggle_live)

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

            common_keys, points1, points2 = common_point_pairs()
            if len(common_keys) < 5:
                QMessageBox.warning(Window_two, "Error", "At least 5 common points required.")
                return

            # 밝기/대비는 보기 보조일 뿐이므로 정합에는 조정 전 원본을 넣는다
            img1 = pixmap_to_bgr(Window_one.viewer.source_pixmap())
            img2 = pixmap_to_bgr(Window_two.viewer.source_pixmap())

            # Live 미리보기와 같은 경로를 쓴다(5쌍 이상이므로 RANSAC 호모그래피)
            matrix, inliers, model_name = estimate_live_transform(
                points1, points2,
                (img1.shape[1], img1.shape[0]), (img2.shape[1], img2.shape[0]))
            base_layer, warped_layer, offset = build_overlay_layers(img1, img2, matrix)

            inlier_mask = None if inliers is None else inliers.ravel()
            canvas_points1 = points1 + [offset[0, 2], offset[1, 2]]
            registered_points2 = transform_points(points2, offset @ matrix)

            overlay_window = RegistrationOverlayWindow()
            overlay_window.default_save_name = (
                f"{Window_two.folder_name}_overlay.png" if Window_two.folder_name else 'overlay.png')
            overlay_window.on_closed = lambda window: (
                active_overlays.remove(window) if window in active_overlays else None)
            overlay_window.set_layers(
                base_layer, warped_layer,
                (canvas_points1, registered_points2, inlier_mask, common_keys),
                status=registration_status(common_keys, canvas_points1, registered_points2,
                                           inlier_mask, model_name),
                origin=(offset[0, 2], offset[1, 2]))
            overlay_window.show()
            active_overlays.append(overlay_window)
        except Exception as e:
            QMessageBox.critical(Window_two, "Error", str(e))

    reg_action.triggered.connect(registration_func)
    update_registration_button()

    sys.exit(app.exec())

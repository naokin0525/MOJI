"""
A custom QWidget that serves as a drawing canvas for creating characters.
"""

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen, QImage, QColor
from PyQt5.QtCore import Qt, QPoint


class DrawingCanvas(QWidget):
    """
    A canvas that captures mouse events to allow free-form drawing.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StaticContents)
        self.setFixedSize(400, 400)

        # The image buffer where the drawing is stored
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)

        self.drawing = False
        self.last_point = QPoint()
        self.strokes = []
        self.current_stroke = []

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            self.current_stroke = [self.last_point]

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.drawing:
            painter = QPainter(self.image)
            pen = QPen(Qt.black, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            painter.end()

            self.last_point = event.pos()
            self.current_stroke.append(self.last_point)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            if self.current_stroke:
                self.strokes.append(self.current_stroke)
            self.current_stroke = []

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(event.rect(), self.image, self.rect())

    def clear_canvas(self):
        """Resets the canvas to a blank state."""
        self.image.fill(Qt.white)
        self.strokes = []
        self.current_stroke = []
        self.update()

    def get_strokes(self) -> list[list[QPoint]]:
        """Returns the collected strokes."""
        return self.strokes

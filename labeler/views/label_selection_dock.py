from PySide.QtCore import *
from PySide.QtGui import *

from .compiled.label_selection_dock_ui import Ui_DockWidget
from labeler.modules.list_labels_model import ListLabelsModel
from labeler.modules.config import Config, TextColorMap


class LabelSelectionDock(QDockWidget, Ui_DockWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        font = self.listView.font()
        # print(font)
        # print(font.pointSize())
        font.setPointSize(int(font.pointSize() * 1.6))
        # print(font.pointSize())
        self.listView.setFont(font)

        self.delegate = QListLabelDelegate(parent=self.listView)
        self.model = ListLabelsModel(self)
        self.listView.setItemDelegate(self.delegate)
        self.listView.setModel(self.model)

        self.listView.setSelectionMode(QAbstractItemView.NoSelection)
        self.listView.setFocusPolicy(Qt.NoFocus)
        self.listView.clicked.connect(self.item_clicked)

        self.current_selection = 0

    def next_selection(self):
        if self.current_selection < self.model.rowCount() - 1:
            self.current_selection += 1
            self.model.dataChanged.emit(
                self.model.index(self.current_selection),
                self.model.index(0)
            )

    def prev_selection(self):
        if self.current_selection > 0:
            self.current_selection -= 1
            self.model.dataChanged.emit(
                self.model.index(self.current_selection),
                self.model.index(0)
            )

    def get_tag(self):
        return list(Config.labels.keys())[self.current_selection]

    @Slot()
    def item_clicked(self, index):
        self.current_selection = index.row()
        self.model.dataChanged.emit(
            self.model.index(self.current_selection),
            self.model.index(0)
        )
        self.parent().apply_tag()

        # print('Clicked: ({}, {})'.format(index.row(), index.column()))


class QListLabelDelegate(QStyledItemDelegate):

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        if self.parent().parent().parent().current_selection == index.row():
            painter.save()

            # Draw selection border
            border_color = TextColorMap.get(Config.labels[index.data()])
            pen = QPen(QColor(border_color))
            # pen = QPen(QColor('#000000'))
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            border_rect = QRectF(option.rect)
            border_rect.adjust(2, 2, -2, -2)
            painter.drawRect(border_rect)

            painter.restore()

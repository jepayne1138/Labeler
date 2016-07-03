from PySide.QtCore import *
from PySide.QtGui import *

from labeler.modules.config import Config, TextColorMap

from .base_model import QListModel


class ListLabelsModel(QListModel):

    def __init__(self, parent=None):
        super().__init__(parent)

        if Config.labels is not None:
            for label, value in Config.labels.items():
                TextColorMap.set(value)
                background_color = QColor(*value)
                background_color.setAlpha(Config.opacity)
                self.append(
                    label, roles={
                        Qt.BackgroundRole: QBrush(background_color),
                        Qt.ForegroundRole: QBrush(QColor(TextColorMap.get(value)))
                    }
                )

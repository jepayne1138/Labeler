from PySide.QtCore import *
from PySide.QtGui import *
# from pprint import pprint

from .base_model import QTableModel


class TableMainModel(QTableModel):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setData(
            self.index(1, 1),
            'The quick brown fox jumped over the lazy dog.'
        )


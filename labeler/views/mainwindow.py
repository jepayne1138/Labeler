from PySide.QtCore import *
from PySide.QtGui import *

from .compiled.mainwindow_ui import Ui_MainWindow
from labeler.models.table_main_model import TableMainModel


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # Create a table model
        self.model = TableMainModel()
        self.delegate = QTestDelegate()
        self.tableMain.setItemDelegate(self.delegate)
        self.tableMain.setModel(self.model)

        self.show()


class QTestDelegate(QStyledItemDelegate):

    def setModelData(self, editor, model, index):
        pass

    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)
        key_filter = KeyPressFilter(self)
        editor.installEventFilter(key_filter)
        return editor


class KeyPressFilter(QObject):

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            return True
        return QObject.eventFilter(self, obj, event)

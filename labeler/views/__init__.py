from PySide.QtCore import *
from PySide.QtGui import *

from .mainwindow import MainWindow


def exec_app(args):
    app = QApplication(args)

    filename, _ = QFileDialog.getOpenFileName(
        caption='Open File',
        filter='JSON Files/ XML Files / Excel Files (*.json *.xml *.xlsx *.xls)'

    )

    if not filename:
        return

    main_cont = MainWindow(filename)
    return app.exec_()

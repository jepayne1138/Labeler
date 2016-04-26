from PySide.QtCore import *
from PySide.QtGui import *

from .mainwindow import MainWindow


def exec_app(args):
    app = QApplication(args)

    main_cont = MainWindow()
    return app.exec_()

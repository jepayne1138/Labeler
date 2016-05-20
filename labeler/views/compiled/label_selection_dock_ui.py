# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:/Users/james.payne/Documents/Python/Labeler/labeler/views/ui/label_selection_dock.ui'
#
# Created: Tue May  3 09:03:52 2016
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_DockWidget(object):
    def setupUi(self, DockWidget):
        DockWidget.setObjectName("DockWidget")
        DockWidget.resize(400, 300)
        DockWidget.setFeatures(QtGui.QDockWidget.DockWidgetFloatable|QtGui.QDockWidget.DockWidgetMovable)
        DockWidget.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea|QtCore.Qt.RightDockWidgetArea)
        self.dockWidgetContents = QtGui.QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.gridLayout = QtGui.QGridLayout(self.dockWidgetContents)
        self.gridLayout.setObjectName("gridLayout")
        self.listView = QtGui.QListView(self.dockWidgetContents)
        self.listView.setObjectName("listView")
        self.gridLayout.addWidget(self.listView, 1, 0, 1, 1)
        self.gridLayoutDisplay = QtGui.QGridLayout()
        self.gridLayoutDisplay.setObjectName("gridLayoutDisplay")
        self.labelCell = QtGui.QLabel(self.dockWidgetContents)
        self.labelCell.setObjectName("labelCell")
        self.gridLayoutDisplay.addWidget(self.labelCell, 1, 0, 1, 1)
        self.lineEditHeader = QtGui.QLineEdit(self.dockWidgetContents)
        self.lineEditHeader.setReadOnly(True)
        self.lineEditHeader.setObjectName("lineEditHeader")
        self.gridLayoutDisplay.addWidget(self.lineEditHeader, 0, 1, 1, 1)
        self.labelHeader = QtGui.QLabel(self.dockWidgetContents)
        self.labelHeader.setObjectName("labelHeader")
        self.gridLayoutDisplay.addWidget(self.labelHeader, 0, 0, 1, 1)
        self.textEditCell = QtGui.QTextEdit(self.dockWidgetContents)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textEditCell.sizePolicy().hasHeightForWidth())
        self.textEditCell.setSizePolicy(sizePolicy)
        self.textEditCell.setMinimumSize(QtCore.QSize(0, 20))
        self.textEditCell.setMaximumSize(QtCore.QSize(16777215, 20))
        self.textEditCell.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textEditCell.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textEditCell.setObjectName("textEditCell")
        self.gridLayoutDisplay.addWidget(self.textEditCell, 1, 1, 1, 1)
        self.gridLayout.addLayout(self.gridLayoutDisplay, 0, 0, 1, 1)
        DockWidget.setWidget(self.dockWidgetContents)

        self.retranslateUi(DockWidget)
        QtCore.QMetaObject.connectSlotsByName(DockWidget)

    def retranslateUi(self, DockWidget):
        DockWidget.setWindowTitle(QtGui.QApplication.translate("DockWidget", "Label Selection", None, QtGui.QApplication.UnicodeUTF8))
        self.labelCell.setText(QtGui.QApplication.translate("DockWidget", "Cell", None, QtGui.QApplication.UnicodeUTF8))
        self.labelHeader.setText(QtGui.QApplication.translate("DockWidget", "Header", None, QtGui.QApplication.UnicodeUTF8))


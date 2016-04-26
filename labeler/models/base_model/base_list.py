from PySide.QtCore import *
from PySide.QtGui import *


class QListModel(QAbstractItemModel):

    """To make the list editable, override flags() to return ItemIsEditable"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = []
        self.headers = {Qt.Horizontal: [QBaseItem(self)], Qt.Vertical: []}

    def index(self, row, column, parent=QModelIndex()):
        return self.createIndex(row, column)

    def parent(self, index):
        return QModelIndex()

    def rowCount(self, parent=QModelIndex()):
        return len(self.values)

    def columnCount(self, parent=QModelIndex()):
        return 1

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        return self.values[index.row()].data(role=role)

    # Editing functions
    def setData(self, index, value, role=Qt.DisplayRole):
        if not index.isValid():
            return False
        self.values[index.row()].setData(value, role)
        self.dataChanged.emit(index, index)
        return True

    # Headers
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        try:
            return self.headers[orientation][section].data(role=role)
        except IndexError:
            return None

    def setHeaderData(self, section, orientation, value, role=Qt.DisplayRole):
        if (section < 0) or (orientation == Qt.Horizontal and section > 0):
            return False
        self.headers[orientation][section].setData(value, role)
        self.headerDataChanged.emit(orientation, section, section)
        return True

    # Adding or removing rows
    def insertRows(self, row, count=1, parent=QModelIndex()):
        # Test that at least one is inserted at an non-negative index
        if (count < 1) or (row < 0):
            return False
        self.beginInsertRows(parent, row, row + count - 1)

        # Insert new QBaseItems into the values list
        for _ in range(count):
            self.values.insert(row, QBaseItem(self))
            # Also insert header rows
            self.headers[Qt.Vertical].insert(row, QBaseItem(self))

        self.endInsertRows()

    def removeRows(self, row, count=1, parent=QModelIndex()):
        # Test that at least one is inserted at an non-negative index
        if (count < 1) or (row < 0):
            return False
        self.beginRemoveRows(parent, row, row + count - 1)

        # Delete from values list
        for _ in range(count):
            del self.values[row]
            # Also delete header rows
            del self.headers[Qt.Vertical][row]

        self.endRemoveRows()

    # Emulate list functions
    def insert(self, index, value, header='', **roles):
        # Make sure that the index is not large than the list
        if index > self.rowCount(index):
            return
        if not self.insertRows(index):
            # Return if the row could not be inserted
            return
        self.setData(index, value)
        self.setRoles(index, **roles)
        self.setHeaderData(index, Qt.Vertical, header)

    def append(self, value, header='', **roles):
        index = self.rowCount()
        self.insert(index, value, header)

    def pop(self, index):
        if -len(self.values) <= index <= (len(self.values) - 1):
            return self.values.pop(index)

    def __len__(self):
        return len(self.values)

    def setRoles(self, index, **roles):
        for role, value in roles.items():
            try:
                role = getattr(Qt, role)
                self.setData(index, value, role)
            except AttributeError:
                pass

    def items(self):
        return self.values

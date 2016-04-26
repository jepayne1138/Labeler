"""TODO:  Basically all documentation"""
from PySide.QtCore import *
from PySide.QtGui import *

from .base_item import QBaseItem


class QTableModel(QAbstractItemModel):

    """To make the table editable, override flags() to return ItemIsEditable"""

    def __init__(self, parent=None, item=None):
        super().__init__(parent)
        self.values = {}
        self.maxRow = 2 ** 20
        self.maxColumn = 2 ** 14
        self.headers = {Qt.Horizontal: [], Qt.Vertical: []}

        # Every blank cell still needs an item, but we don't want to have to
        # keep making new empty items, therefore we create a singular
        # reference to an empty display item to be used for all empty cells
        self.item = item if item is not None else QBaseItem
        self.defaultEmpty = self.item()

    def index(self, row, column, parent=QModelIndex()):
        return self.createIndex(row, column)

    def parent(self, index):
        return QModelIndex()

    def rowCount(self, parent=QModelIndex()):
        return self.maxRow

    def columnCount(self, parent=QModelIndex()):
        return self.maxColumn

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if (index.row() in self.values and
                index.column() in self.values[index.row()]):
            return self.values[index.row()][index.column()].data(role=role)
        else:
            return self.defaultEmpty.data(role=role)

    # Editing functions
    def setData(self, index, value, role=Qt.DisplayRole):
        self._setData(index, value, role)

        # Pack the values dict
        self._packValues(index)

        # Emit dataChangd signal
        self.dataChanged.emit(index, index)

        return True

    def _setData(self, index, value, role):
        """Actually sets the data and can be used by other methods

        IMPORTANT: Any method that calls this method MUST call
            self.dataChanged.emit(index, index) afterwards
        """
        if not index.isValid():
            return False

        # Make sure the row entry exists
        if index.row() not in self.values:
            self.values[index.row()] = {}

        # Create a new item if non exists at the given index
        if index.column() not in self.values[index.row()]:
            self.values[index.row()][index.column()] = self.item(self)

        self.values[index.row()][index.column()].setData(value, role)

    def _packValues(self, index):
        """Checks if the item is empty and packs the values dictionary

        First checks if the item at the given index is equal to the
        default empty object, and if so removes it from the values dict.
        Then, if it did remove an object, we also need to check if the dict
        of column values for that row is now also empty so we can remove
        that as well.
        """
        # Check for empty item at the current index to pack dict
        if self.values[index.row()][index.column()] == self.defaultEmpty:
            del self.values[index.row()][index.column()]
            if len(self.values[index.row()]) == 0:
                del self.values[index.row()]

    # Headers
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """Returns the header data if it exists else row and column indices"""
        try:
            return self.headers[orientation][section].data(role=role)
        except IndexError:
            if role == Qt.DisplayRole:
                return section + 1
            return None

    def setHeaderData(self, section, orientation, value, role=Qt.DisplayRole):
        if section < 0:
            return False

        if self.headerData(section, orientation) is None:
            self.headers[orientation][section] = QBaseItem

        self.headers[orientation][section].setData(value, role)
        self.headerDataChanged.emit(orientation, section, section)
        return True

    def flags(self, index):
        return super().flags(index) | Qt.ItemIsEditable

    def setRoles(self, index, roles=None):
        """Sets roles for a given index

        Args:
            index (QModelIndex): Index of the item to set roles for
            roles (Optional[Dict[int, RoleValue]]): Dictionary of
                Qt.ItemDataRoles and their corresponding values.
                Defaults to None.
        """
        if roles is not None:
            for role, value in roles.items():
                self._setData(index, value, role)

        # Pack the values dict
        self._packValues(index)

        # Emit dataChangd signal
        self.dataChanged.emit(index, index)

    def items(self):
        return self.values

    # def pack(self):
    #     """Removes any empty cells from the underlying data structure"""
    #     for row_index, row_dict in self.values.items():
    #         for key, value in row_dict.items():
    #             if value == self.defaultEmpty:
    #                 del self.values[row_index][key]
    #         if row_dict == {}:
    #             del self.values[row_index]

"""TODO:  Module level documentation"""
from PySide.QtCore import *
from PySide.QtGui import *


class QBaseItem(QStandardItem):

    """A base class for handling items in QTableModels and QListModels"""

    def __init__(self, display='', roles=None, clear_flags=False, flags=0):
        """Initializes a new QBaseItem

        Args:
            display (Optional[str]): Value to be displayed for this item.
                Defaults to the empty string ''.
            roles (Optional[Dict[int, RoleValue]]): Dictionary of
                Qt.ItemDataRoles and their corresponding values.
                Defaults to None.
            clear_flags (Optional[bool]):  Clears all the default flags for
                a QStandardItem before setting the user defined flags.
                Defaults to False.
            flags (Optional[int]): Argument should be any number of
                Qt.ItemFlags bitwise ORed together. This resulting value is
                ORed with the current flags value. Defaults to 0.
        """
        super().__init__()
        # Set the display value
        self.setData(display, role=Qt.DisplayRole)
        # Set any given Qt.ItemDataRoles
        if roles is not None:
            for role, value in roles.items():
                self.setData(value, role=role)
        # Check to see if we want to clear any existing flags
        if clear_flags:
            self.setFlags(Qt.NoItemFlags)
        # Set any given Qt.ItemFlags
        self.setFlags(self.flags() | flags)  # flags defaults to 0

    # def setData(self, value, role=Qt.UserRole + 1):
    #     # We set the size hint for the Qt.DisplayRole
    #     if role == Qt.DisplayRole:
    #         fm = QFontMetrics(self.font())
    #         self.setSizeHint(QSize(fm.width(str(value)), fm.height()))
    #     super().setData(value, role)

    def type(self):
        """Returns a type distinguishing the custom item from the base class

        When subclassing we should return a value greater than or equal to
        QStandardItem.UserType

        Returns:
            int: The integer representing the custom type of this class
        """
        return QStandardItem.UserType + 1

    # def __eq__(self, obj):
    #     """Checks for equality of items base on the values of Qt.DisplayRole

    #     Returns:
    #         bool: True if the given object has the same Qt.DisplayRole value
    #             as this object, False otherwise or if the given object does
    #             not support the data method or role keyword of said method.
    #     """
    #     try:
    #         return self.data(role=Qt.DisplayRole) == obj.data(role=Qt.DisplayRole)
    #     except (AttributeError, TypeError):
    #         return False

    def __eq__(self, obj):
        """Checks for equality with the object on all Qt.ItemDataRoles"""
        try:
            for role in Qt.ItemDataRole.values.values():
                if self.data(role=role) != obj.data(role=role):
                    return False
            return True
        except (AttributeError, TypeError):
            return False

    def __repr__(self):
        return '<QBaseItem(DisplayRole={})>'.format(
            self.data(role=Qt.DisplayRole)
        )

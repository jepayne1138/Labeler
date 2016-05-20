from PySide.QtCore import *
from PySide.QtGui import *
# from pprint import pprint

from .base_model import QTableModel


class TableMainModel(QTableModel):

    def __init__(self, tag_manager, parent=None):
        """Wraps a TagManager instance in a QTableModel

        Args:
            tag_manager (TagManager): The TagManager instance to be handled
        """
        super().__init__(parent)
        self.tm = tag_manager

        # Initialize the values dictionary
        for row in self.tm.rows():
            for col in self.tm.columns(row):
                self.setData(
                    self.index(row, col),
                    self.tm.get(row, col).html()
                )
                # Set plain text as user role for size hint
                self.setData(
                    self.index(row, col),
                    self.tm.get(row, col).plain_text(),
                    role=Qt.UserRole + 1
                )

        # Set headers
        if self.tm.headers:
            for index, header in self.tm.headers.items():
                self.setHeaderData(index, header)

        # with open('dump.txt', 'w') as dump:

        #     for row in self.tm.rows():
        #         for col in self.tm.columns(row):
        #             item = self.getItem(row, col)
        #             # print(item.data(role=Qt.DisplayRole))
        #             dump.write(
        #             # print(
        #                 '[{: >3}] - ({: >3}, {: >3}): {}\n'.format(
        #                     item.sizeHint().width(),
        #                     row, col,
        #                     str(item.data(role=Qt.DisplayRole))
        #                 )
        #             )

        # self.setData(
        #     self.index(0, 0),
        #     self.tm.get(0, 0).highlight(1)
        # )

    def row_col_iterator(self, *args, **kwargs):
        for index in self.tm.index_iterator(*args, **kwargs):
            yield index
        raise StopIteration()

    def index_iterator(self, *args, **kwargs):
        """Generic parameters as we're just mirroring the main method in
        a different scoper"""
        for row, col in self.tm.index_iterator(*args, **kwargs):
            yield self.index(row, col)
        raise StopIteration()

    def flags(self, index):
        return Qt.NoItemFlags | Qt.ItemIsEnabled
        # return (QAbstractItemModel.flags(index) & Qt.NoItemFlags)

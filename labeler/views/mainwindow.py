from pprint import pprint
from inspect import getmembers
from collections import defaultdict
import os

from PySide.QtCore import *
from PySide.QtGui import *

from .compiled.mainwindow_ui import Ui_MainWindow
from .label_selection_dock import LabelSelectionDock
from labeler.models.table_main_model import TableMainModel
from labeler.models.tag_manager import TagManager


# For test purposes only

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, filename, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.modified = False
        self.work_dir, self.o_filename = os.path.split(filename)
        self.basename, self.extension = os.path.splitext(self.o_filename)

        # Random stuff
        if self.extension == '.xml':
            self.filename = filename
            self.tm = TagManager.from_xml(filename)
        elif self.extension == '.json':
            self.filename = filename
            self.tm = TagManager.from_json(filename)
        else:
            self.filename = None
            self.tm = TagManager.from_xlsx(filename)

        self.labelDock = LabelSelectionDock(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.labelDock, Qt.Vertical)

        # Create a table model
        self.model = TableMainModel(self.tm, self)
        self.delegate = QTestDelegate(parent=self.tableMain)
        self.tableMain.setSelectionMode(QAbstractItemView.NoSelection)
        self.tableMain.setItemDelegate(self.delegate)
        self.tableMain.setModel(self.model)

        # # Adjust the column widths
        col_widths = max_column_size_hints(self.model, self.tableMain)
        for column, width in col_widths.items():
            self.tableMain.setColumnWidth(column, width)

        # Select the current first untagged word
        self.cur_word = self.tm.next_untagged()
        if not self.cur_word:
            self.cur_word = self.tm.next_untagged(allow_tagged=True)
        self.update_dock()
        self.highlight_current()
        self.tableMain.scrollTo(
            self.model.index(self.cur_word.row, self.cur_word.column)
        )

        # Install key event filter
        self.tableMain.installEventFilter(TabelKeyFilter(self))
        self.labelDock.installEventFilter(TabelKeyFilter(self))

        self.actionSave.triggered.connect(self.save)
        self.actionSave_As.triggered.connect(self.save_as)
        self.actionCount_Labels.triggered.connect(self.count_labels)

        self.showMaximized()

    # def select(self):
    #     self.cur_cell.

    def get_next(self):
        word = self.cur_word.get_next()
        if word is not None:
            # Clear highlighting from previous if cell change
            if self.cur_word.row != word.row or self.cur_word.column != word.column:
                self.model.setData(
                    self.model.index(self.cur_word.row, self.cur_word.column),
                    self.cur_word.parent.html()
                )
            # Highlight the new word
            self.cur_word = word
            self.highlight_current()
            self.update_dock()
            self.tableMain.scrollTo(
                self.model.index(self.cur_word.row, self.cur_word.column)
            )

    def get_prev(self):
        word = self.cur_word.get_prev()
        if word is not None:
            # Clear highlighting from previous if cell change
            if self.cur_word.column != word.column:
                self.model.setData(
                    self.model.index(self.cur_word.row, self.cur_word.column),
                    self.cur_word.parent.html()
                )
            # Highlight the new word
            self.cur_word = word
            self.highlight_current()
            self.update_dock()
            self.tableMain.scrollTo(
                self.model.index(self.cur_word.row, self.cur_word.column)
            )

    def up_cell(self):
        print('Not implemented yet:  Go up cells')

    def down_cell(self):
        print('Not implemented yet:  Go down cells')

    def highlight_current(self):
        self.model.setData(
            self.model.index(self.cur_word.row, self.cur_word.column),
            self.cur_word.parent.highlight(self.cur_word.index)
        )

    def update_dock(self):
        header = self.model.headerData(self.cur_word.column)
        self.labelDock.lineEditHeader.setText(str(header))
        self.labelDock.textEditCell.setHtml(
            self.cur_word.parent.highlight(self.cur_word.index)
        )

    def apply_tag(self):
        # Currently still stays modify if you pick same tags over again
        # could fix this later if I want
        self.modified = True
        tag = self.labelDock.get_tag()

        # Apply the tag
        self.cur_word.add_tag(tag)

        # Advance the word
        self.get_next()

    @Slot()
    def count_labels(self, event=None):
        self.tm.count_labels()

    @Slot()
    def save(self, event=None):
        if not self.filename:
            return self.save_as()
        else:
            extension = os.path.splitext(self.filename)[1]
            if extension == '.json':
                with open(self.filename, 'w') as save_file:
                    self.tm.write_json(save_file)
            elif extension == '.xml':
            # print('Saving to: {}'.format(self.filename))
                with open(self.filename, 'wb') as save_file:
                    self.tm.write_xml(save_file)
            self.modified = False
            return True
        #     base_filename = os.path.splitext(self.tm.basename)[0]
        # print(base_filename)

    @Slot()
    def save_as(self, event=None):
        """Returns False on cancelation"""
        self.filename, _ = QFileDialog.getSaveFileName(
            self, 'Save As', os.path.join(self.work_dir, self.basename),
            'JSON File (*.json)'
            # 'XML File (*.xml);;JSON File (*.json)'  # (Disabled in favor of safer JSON)
        )
        if self.filename:
            return self.save()
        return False

    @Slot()
    def close(self, event=None):
        if self.modified_prompt_accept_close():
            super().close()


    def closeEvent(self, event):
        if self.modified_prompt_accept_close():
            event.accept()
        else:
            event.ignore()

    def modified_prompt_accept_close(self):
        if self.modified:
            # Prompt for save
            reply = QMessageBox.question(
                self, 'Save Confirmation',
                'The file has been modified.\nDo you want to save your changes?',
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )

            if reply == QMessageBox.Save:
                if not self.save():
                    return False
            elif reply == QMessageBox.Cancel:
                return False
        return True



def max_column_size_hints(table_model, table_view):
    """
    Ideally I subclass the TableView and make this a method of that

    Args:
        table_model (QTableModel): A QTableModel instance

    Returns:
        Dict[int, int]: Column numbers and corresponding max sizeHints
    """
    max_columns = defaultdict(int)
    delegate = table_view.itemDelegate()
    dummy_options = QStyleOptionViewItem()
    for index in table_model.index_iterator():
        size = delegate.sizeHint(dummy_options, index)
        max_columns[index.column()] = max(
            max_columns[index.column()], size.width()
        )
    return max_columns


class QTestDelegate(QStyledItemDelegate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def paint(self, painter, option, index):
        options = QStyleOptionViewItemV4(option)
        self.initStyleOption(options, index)

        # Remove focuss
        options.state = options.state & ~QStyle.State_HasFocus

        # pprint(getmembers(options))
        if options.widget is None:
            style = QApplication.style()
        else:
            style = options.widget.style()

        doc = QTextDocument()
        doc.setHtml(options.text)

        options.text = ""
        style.drawControl(QStyle.CE_ItemViewItem, options, painter)

        ctx = QAbstractTextDocumentLayout.PaintContext()

        # Highlighting text if item is selected
        textRect = style.subElementRect(QStyle.SE_ItemViewItemText, options)
        painter.save()
        painter.translate(textRect.topLeft())
        painter.setClipRect(textRect.translated(-textRect.topLeft()))
        doc.documentLayout().draw(painter, ctx)

        painter.restore()

    def sizeHint(self, option, index):
        font_metrics = option.fontMetrics
        text = index.model().data(index, role=Qt.UserRole + 1)
        doc = QTextDocument(text)
        doc.setDefaultFont(option.font)
        return QSize(doc.idealWidth(), font_metrics.height())

    # def paint(self, painter, option, index):
    #     # new_option = option
    #     # new_option.state = option.state & ~QStyle.State_HasFocus
    #     # super().paint(painter, new_option, index)
    #     if index.data():
    #         print(index.data())
    #     super().paint(painter, option, index)

    def setModelData(self, editor, model, index):
        pass

    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)
        editor.installEventFilter(KeyPressFilter(self))
        # print(editor)
        return editor

    # def editorEvent(self, event, model, option, index):
    #     """Called when beginning the editing event"""
    #     print('\nEditorEvent:')
    #     print('  event  : {}'.format(event))
    #     print('  model  : {}'.format(model))
    #     print('  option : {}'.format(option))
    #     print('  index  : {}'.format(index))
    #     super().editorEvent(event, model, option, index)


class KeyPressFilter(QObject):

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            return True
        # elif event.type() == QEvent.CursorChange:
        #     print('cursor moved')
        # elif event.type() == QEvent.MouseButtonPress:
        #     print('mouse pressed')
        return QObject.eventFilter(self, obj, event)


# Handle key presses (in a totally stupid way)


def _get_next_word(parent, obj, event):
    parent.parent().get_next()


def _get_prev_word(parent, obj, event):
    parent.parent().get_prev()

def _go_up_cell(parent, obj, event):
    parent.parent().up_cell()

def _go_down_cell(parent, obj, event):
    parent.parent().down_cell()


def _get_next_label(parent, obj, event):
    main_window = parent.parent()
    main_window.labelDock.next_selection()


def _get_prev_label(parent, obj, event):
    main_window = parent.parent()
    main_window.labelDock.prev_selection()


def _apply_tag(parent, obj, event):
    main_window = parent.parent()
    main_window.apply_tag()


class TabelKeyFilter(QObject):

    """Currently storing all mapped keys to functions in a dict so that I can
    easily only filter the key presses that have some mapped function. I want
    to filter all keys, I can still do that with this setup.
    """

    nav = {
        Qt.Key_Right: _get_next_word,
        Qt.Key_Left: _get_prev_word,
        Qt.Key_Down: _get_next_label,
        Qt.Key_Up: _get_prev_label,
        Qt.Key_Space: _apply_tag,
    }
    nav.update({Qt.Key_Up: _go_up_cell, Qt.Key_Down: _go_down_cell})

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.KeyPress and
                event.key() in TabelKeyFilter.nav.keys()):
            # Handle key events
            TabelKeyFilter.nav[event.key()](self, obj, event)
            return True

        return QObject.eventFilter(self, obj, event)


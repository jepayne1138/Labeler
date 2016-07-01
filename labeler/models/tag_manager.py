"""Despite the misleading module name, XML is a bad idea as I don't quote
the values in between tagged words, meaning we could format the XML
differently and still have well-formed data but that would be incorrect.
Therefore, while the XML opening and saving works, I have 'disabled' saving
as XML in favor of JSON, despite the slightly larger file sizes (I simply
removed the XML option from the save filter in the save dialog in the
mainwindow.py file)."""

import os
import re
import copy
import xml.etree.ElementTree as ET
import json
import xlrd
import math
import collections
from labeler.models.config import Config, TextColorMap


XLSX_ENCODING = "cp1252"
ENCODING = 'utf-8'
XML_VERSON = '1.0'
HEADER = 'header'
CONTENT = 'content'
FILE = 'file'
COLUMN_TAG = 'c'
ROW_TAG = 'r'

RE_WORD = re.compile("([A-Za-z0-9]+)")


class TableDictFactory:

    """Factory for creating nested dicts of possible sparse tabular data"""

    @classmethod
    def from_xml(cls, filename):
        """Loads from an existing tag XML file"""
        # Table factory handles packing the tabular data nicely for us,
        # we just need to iterate over the data for it
        table_factory = cls()

        tree = ET.parse(filename)

        root = tree.getroot()

        # Parse headers if they are in the file
        headers = root.find(HEADER)
        if headers:
            for column in headers.findall(COLUMN_TAG):
                try:
                    table_factory.add_header(
                        int(column.attrib['value']),
                        column.text,
                    )
                except KeyError:
                    pass

        # Iterate over rows and columns to import the data and tags
        for row in root.findall(ROW_TAG):
            for col in row.findall(COLUMN_TAG):
                table_factory.add_value(
                    int(row.attrib['value']),
                    int(col.attrib['value']),
                    cls.xml_parse_words(col)
                )

        return table_factory.generate()

    @classmethod
    def from_json(cls, filename):
        """Loads from an existing tag JSON file"""
        table_factory = cls()

        with open(filename, 'r') as json_file:
            root = json.load(json_file)

        # Parse headers if they are in the file
        if HEADER in root:
            for column, head_dict in root[HEADER].items():
                try:
                    table_factory.add_header(
                        int(column),
                        head_dict['value'],
                    )
                except KeyError:
                    pass

        # Iterate over rows and columns to import the data and tags
        for row, column_dict in root[CONTENT].items():
            for col, content in column_dict.items():
                table_factory.add_value(
                    int(row), int(col),
                    cls.json_parse_words(content)
                )

        return table_factory.generate()

    @classmethod
    def from_xlsx(cls, filename, headers=True, sheet=0):
        """Reads an xlsx file and returns a dictionary of data

        If headers is true, key in the returned dict will give a dictionary of
        the headers
        """
        xl_book = xlrd.open_workbook(filename, encoding_override=XLSX_ENCODING)
        xl_sheet = xl_book.sheet_by_index(sheet)

        # Table factory handles packing the tabular data nicely for us,
        # we just need to iterate over the data for it
        table_factory = cls()

        # Populate and attach the headers dict
        if headers:
            for column, header in enumerate(xl_sheet.row_values(0)):
                table_factory.add_header(column, header)

        # Get values
        start_row = 1 if headers else 0
        for row in range(start_row, xl_sheet.nrows):
            columns = xl_sheet.row_values(row)
            for column, value in enumerate(columns):
                coded_value = cls.normalize(value)
                table_factory.add_value(row, column, coded_value)

        # Return the dict
        return table_factory.generate()

    @staticmethod
    def normalize(input_var):
        """ Normalizes a variable as string, cleans and converts float to int """
        var = input_var
        try:
            var = var.encode(XLSX_ENCODING).decode(ENCODING, 'ignore')
        except AttributeError:
            try:
                var = float(var)
                if math.isnan(var):
                    return str(input_var)
                if math.isinf(var):
                    return str(input_var)
                if int(var) == var:
                    var = str(int(var))
                else:
                    var = str(var)
            except ValueError:
                pass
        return var

    def __init__(self):
        self.data = {}

    @staticmethod
    def xml_parse_words(element):
        split = []
        if element.text:
            split.append(element.text)
        for word in element.iter('w'):
            split.append(TempWord(word.text, word.attrib['tags']))
            if word.tail:
                split.append(word.tail)
        return split

    @staticmethod
    def json_parse_words(json_list):
        return [
            TempWord(word['word'], word['tags'])
            if isinstance(word, dict) else word
            for word in json_list
        ]

    def add_header(self, column, value):
        # Check that a value is not empty
        if value == '':
            return

        # Check that the header dict exists
        if HEADER not in self.data:
            self.data[HEADER] = {}

        if column not in self.data[HEADER]:
            self.data[HEADER][column] = {
                'value': '',
                'tags': []
            }

        self.data[HEADER][column]['value'] = value

    def add_value(self, row, column, value):
        # Check that a value is not empty
        if value == '':
            return

        # Check that the row dict exists
        if row not in self.data:
            self.data[row] = {}

        self.data[row][column] = value

    def generate(self):
        return copy.deepcopy(self.data)


class TempWord:

    def __init__(self, text, tags):
        """Tags takes either space separated or an iterable"""
        self.text = text
        if tags:
            try:
                self.tags = set(tags.split())
            except AttributeError:
                self.tags = set(tags)
        else:
            self.tags = None

# --------------------------------------------------------------------------
# Tag Manager and supporting classes
# --------------------------------------------------------------------------


class TagWord:

    tag = 'w'  # Defaults to short 'w' for 'word' as there will be many
    hightlight_css = 'background-color: black; color: white;'

    @classmethod
    def is_inst(cls, obj):
        return isinstance(obj, cls)

    def __init__(self, parent, word, row, column, index, tags=None):
        self.parent = parent
        self.word = word
        self.row = row
        self.column = column
        self.index = index
        self.tags = set() if tags is None else set(tags)
        self.tag_probabilities = None

    def __str__(self):
        return self.word

    def __repr__(self):
        return '<TagWord(word="{word}", tags={tags})>'.format(**self.__dict__)

    def html(self):
        if len(self.tags) == 0:
            return self.word
        r, g, b, a = Config.labels[next(iter(sorted(self.tags)))]
        # background_color = (r, g, b, '{}%'.format(int(a * 100)))
        return self.highlight(
            'background-color: rgba({}, {}, {}, {}); color: {};'.format(
                r, g, b, '{}%'.format(int(a * 100)),
                TextColorMap.get((r, g, b, a))
            )
        )

    def get_next(self):
        return self.parent.get_next(self.index)

    def get_prev(self):
        return self.parent.get_prev(self.index)

    def highlight(self, style=None):
        if style is None:
            style = TagWord.hightlight_css
        return '<span style="{tags}">{word}</span>'.format(
            tags=style,
            word=self.word,
        )

    def xml(self):
        ret_element = ET.Element(TagWord.tag, {'tags': ' '.join(self.tags)})
        ret_element.text = self.word
        return ret_element
        # return '<{xml_tag} tags="{tags}">{word}</{xml_tag}>'.format(
        #     xml_tag=TagWord.tag,
        #     tags=' '.join(self.tags),
        #     word=self.word,
        # )

    def json(self):
        """Returns a dict representing the word information for JSON"""
        return {
            'tags': list(self.tags),
            'word': self.word,
        }

    def add_tag(self, tag):
        """Adds a tag to the tag set"""
        # Currently only going to allow one
        self.tags = set((tag,))
        # self.tags.add(tag)

    def remove_tag(self, tag):
        """Removes a tag from the tag set

        Returns:
            bool: True if the tag was successfully removed

        """
        try:
            self.tags.remove(tag)
            return True
        except KeyError:
            return False


class TagCell:

    """A cell has row and column index number, and some string value
    Each word in the string is represented by a TagWord instance
    and can be assigned one or more tags
    """

    def __init__(self, parent, string, row, column, from_type):
        self.parent = parent
        self.string = string
        self.row = row
        self.column = column

        if from_type != 'xlsx':
            self.split = self.split_tempwords()
            self.string = self.plain_text()
        else:
            # For Excel (xlsx) or other non-tagged plain string imports
            self.split = self.split_string()

    def split_string(self):
        word_index = 0
        split = []
        for word in RE_WORD.split(self.string):
            if word != '':
                if RE_WORD.match(word):
                    split.append(
                        TagWord(self, word, self.row, self.column, word_index)
                    )
                    word_index += 1
                else:
                    split.append(word)
        return split

    def split_tempwords(self):
        word_index = 0
        split = []
        for word in self.string:
            if isinstance(word, TempWord):
                tag_word = TagWord(
                    self, word.text,
                    self.row, self.column,
                    word_index, tags=word.tags
                )
                split.append(tag_word)

                word_index += 1
            else:
                split.append(word)
        return split

    def iter_word_strings(self):
        for word in self.split:
            if isinstance(word, TagWord):
                yield word
        return

    def __str__(self):
        return self.string

    def __repr__(self):
        return '<TagCell(string="{string}", row={row}, column={column}, split={split})>'.format(**self.__dict__)

    def html(self):
        join_list = []
        for word in self.split:
            if TagWord.is_inst(word):
                join_list.append(word.html())
            else:
                join_list.append(str(word))
        return ''.join(join_list)

    def plain_text(self):
        return ''.join([str(word) for word in self.split])

    def get_next(self, index):
        next_index = self._map_tag_word(index + 1)
        if next_index is not None:
            return self.split[next_index]
        else:
            # Recursively find the next (will have recursion error if we have
            # more than 50 cells in a row with data but no words)
            next_cell = self.parent.get_next_cell(self.row, self.column)
            if next_cell is not None:
                return next_cell.get_next(-1)
            return None

    def get_prev(self, index):
        if index == -1:
            index = len(list(filter(TagWord.is_inst, self.split)))
        prev_index = self._map_tag_word(index - 1)
        if prev_index is not None:
            return self.split[prev_index]
        else:
            # Recursively find the next (will have recursion error if we have
            # more than 50 cells in a row with data but no words)
            prev_cell = self.parent.get_next_cell(self.row, self.column, reverse=True)
            if prev_cell is not None:
                return prev_cell.get_prev(-1)
            return None
            # return self.parent.get_next_cell(self.row, self.column, reverse=True).get_prev(-1)

    def highlight(self, index):
        """Highlights the given word by wrapping in styling tags

        Index is the TagWord index, ie index=n highlights the nth TagWord,
        not the nth word in the split list

        Returns:
            string: Constructed string with html tags and highlighting style
            bool: True in index was in range to highlight
        """
        # Get the real index of the given TagWord index
        split_index = self._map_tag_word(index)

        # If the index was invalid, just return the string
        if split_index is None:
            return self.string

        # Otherwise wrap the TagWord in it's highlight tags
        join_list = []
        for index, word in enumerate(self.split):
            if TagWord.is_inst(word):
                if index == split_index:
                    join_list.append(word.highlight())
                else:
                    join_list.append(word.html())
            else:
                join_list.append(str(word))
        return ''.join(join_list)

    def add_tag(self, index, tag):
        # Get the real index of the given TagWord index
        split_index = self._map_tag_word(index)
        if split_index is not None:
            self.split[split_index].add_tag(tag)

    def remove_tag(self, index, tag):
        # Get the real index of the given TagWord index
        split_index = self._map_tag_word(index)
        if split_index is not None:
            return self.split[split_index].remove_tag(tag)
        return False

    def first_untagged(self, allow_tagged=False):
        """Gets the first untagged TagWord in the instance

        Returns:
            TagWord: TagWord instance if untagged, or None if all tagged
        """
        for i, word in enumerate(self.split):
            if TagWord.is_inst(word):
                if allow_tagged or not word.tags:
                    return word
        return None

    def _map_tag_word(self, index):
        """Given TagWord index n, map index to the full split list"""
        # Check that the given index is in the valid range
        if index not in range(len(list(filter(TagWord.is_inst, self.split)))):
            return None
        for i, word in enumerate(self.split):
            if TagWord.is_inst(word):
                if index == 0:
                    return i
                index -= 1
        return None

    def xml_element(self, tag, attrib=None):
        ret_element = ET.Element(tag, {} if attrib is None else attrib)
        last_element = None
        for item in self.split:
            if TagWord.is_inst(item):
                # This item is a TagWord and we need to add the element
                word_element = item.xml()
                ret_element.append(word_element)
                # Save a reference to last element for tailing further text
                last_element = word_element
            else:
                # The item is not a TagWord, so we just append the string
                if last_element is None:
                    # No TagWord yet to tail text to, so we concatenate with
                    # the root return element
                    if ret_element.text is None:
                        ret_element.text = ''
                    ret_element.text += str(item)
                else:
                    # If a TagWord was already added, we tail the text after
                    # the last element appended
                    if last_element.tail is None:
                        last_element.tail = ''
                    last_element.tail += str(item)
        return ret_element

    def json_element(self):
        """Returns a list intended to be used to represent cell in JSON"""
        ret_list = []
        for item in self.split:
            if TagWord.is_inst(item):
                # This item is a TagWord and we need to add the element
                ret_list.append(item.json())
            else:
                # The item is not a TagWord, so we just append the string
                ret_list.append(str(item))
        return ret_list

    def label_count(self):
        ret_counts = collections.defaultdict(int)
        for item in self.split:
            if TagWord.is_inst(item):
                # This item is a TagWord and we need to add the element
                for tag in item.tags:
                    ret_counts[tag] += 1
        return ret_counts


class TagManager:

    """Manages a list with tagged words"""

    @classmethod
    def from_xlsx(cls, filename, headers=True, sheet=0):
        data = TableDictFactory.from_xlsx(filename, headers, sheet)
        return cls(data, filename, from_type='xlsx')

    @classmethod
    def from_xml(cls, filename):
        data = TableDictFactory.from_xml(filename)
        return cls(data, filename, from_type='xml')

    @classmethod
    def from_json(cls, filename):
        data = TableDictFactory.from_json(filename)
        return cls(data, filename, from_type='json')

    def __init__(self, data, filename='', from_type='json'):
        # Get headers
        self.headers = data[HEADER] if HEADER in data else None
        self.basename = os.path.basename(filename)
        self.from_type = from_type

        # print(self.headers)
        # from pprint import pformat
        # with open('dump.txt', 'w') as dump:
        #     dump.write(pformat(data))

        # Get data
        self.parsed = {}
        for row, row_values in data.items():
            if row != HEADER:
                adjusted_row = row if (self.headers is None or from_type != 'xlsx') else row - 1
                self.parsed[adjusted_row] = {
                    col: TagCell(self, value, adjusted_row, col, from_type=self.from_type)
                    for col, value in row_values.items()
                }

        # from pprint import pformat
        # with open('dump.txt', 'w') as d:
        #     d.write(pformat(self.parsed))


        # self.parsed = {
        #     (row if self.headers is None else row - 1):
        #     {col: TagCell(self, value, row, col) for col, value in row_values.items()}
        #     for row, row_values
        #     in data.items()
        #     if row != HEADER
        # }

    def next_untagged(self, start=None, allow_tagged=False):
        """Get the first cell in natural order with an untagged word

        Args:
            start_tuple (Tuple[int, int]): Row and column index for start

        Returns:
            TagCell: First TagCell after start point in natural order that
                contains at least one untagged word
        """
        # Set starting point
        start_row = 0
        start_col = 0
        if start is not None:
            try:
                start_row = start[0]
                start_col = start[1]
            except (IndexError, KeyError):
                pass

        for row in self.rows(start=start_row):
            if row != start_row:
                start_col = 0
            for col in self.columns(row, start=start_col):
                tag_cell = self.get(row, col)
                word = tag_cell.first_untagged(allow_tagged=allow_tagged)
                if word is not None:
                    return word
        return None

    def get(self, row, column):
        """Returns TagCell object or None if nothing at coordinates"""
        try:
            a = self.parsed[row][column]
            # print('TagManager.get({}, {}) = {}'.format(row, column, a))
            return a
        except KeyError:
            return None

    def headers(self):
        """Creates a list over all headers"""
        return self.headers

    def get_next_cell(self, row, col, reverse=False):
        """Given index for a row and column, return next

        reverse parameter gets previous cell
        """
        index_gen = self.index_iterator(row, col, reverse)
        try:
            next(index_gen)  # Starts by getting current cell
            next_row, next_col = next(index_gen)
            return self.get(next_row, next_col)
            # return self.get(*next(index_gen))
        except StopIteration:
            return None

    def rows(self, start=None, reverse=False):
        # """Creates a generator over all rows with data

        # Each row instance is a generator for all columns in that cell
        # """

        # Set start
        if start is None:
            if reverse:
                start = max(set(self.parsed.keys()))
            else:
                start = 0

        if not reverse:
            return sorted([key for key in self.parsed.keys() if key >= start])
        else:
            return sorted(
                [key for key in self.parsed.keys() if key <= start],
                reverse=True
            )
        # for row in used_rows:
        #     yield self.row_values(row)
        # raise StopIteration()

    def columns(self, row, start=None, reverse=False):

        try:
            # Set start
            if start is None:
                if reverse:
                    start = max(set(self.parsed[row].keys()))
                else:
                    start = 0

            if not reverse:
                return sorted(
                    [key for key in self.parsed[row].keys() if key >= start]
                )
            else:
                return sorted(
                    [key for key in self.parsed[row].keys() if key <= start],
                    reverse=True
                )
        except KeyError:
            return []

    def index_iterator(self, start_row=None, start_col=None, reverse=False):
        """Returns row and column index tuple in natural order"""
        # print('start_row: {}'.format(start_row))
        # print('start_col: {}'.format(start_col))
        # print('reverse: {}'.format(reverse))
        for row in self.rows(start=start_row, reverse=reverse):
            # print('row: {}'.format(row))
            if start_row is not None and row != start_row:
                start_col = None
            for column in self.columns(row, start=start_col, reverse=reverse):
                # print('yielding: ({}, {})'.format(row, column))
                yield (row, column)
        raise StopIteration()

    def row_values(self, row):
        try:
            for column in sorted(self.parsed[row].keys()):
                yield self.parsed[row][column]
            raise StopIteration()
        except KeyError:
            raise StopIteration()

    def string_xml(self):
        return ET.tostring(self.generate_xml(), encoding=ENCODING)

    def write_xml(
            self, file_obj, encoding=ENCODING,
            xml_declaration=True, **kwargs):
        tree = ET.ElementTree(self.generate_xml())
        return tree.write(
            file_obj, encoding=encoding,
            xml_declaration=xml_declaration, **kwargs
        )

    def generate_xml(self):
        """Returns the parsed data with tags as and XML tagged document"""
        # TODO: Implement. We can get an XML Element from each TagCell, now
        # we need to make the whole XML tree.
        root = ET.Element(FILE, attrib={'filename': self.basename})
        root.set('version', XML_VERSON)

        if self.headers:
            header_element = ET.SubElement(root, HEADER)
            for column, header in self.headers.items():
                head_element = ET.SubElement(
                    header_element, COLUMN_TAG, {'value': str(column)}
                )
                head_element.text = header

        for row in sorted(self.parsed.keys()):
            row_element = ET.SubElement(root, ROW_TAG, {'value': str(row)})
            for column in sorted(self.parsed[row].keys()):
                cell_element = self.parsed[row][column].xml_element(
                    COLUMN_TAG, {'value': str(column)}
                )
                row_element.append(cell_element)
        return root

    def write_json(
            self, file_obj, encoding=ENCODING, **kwargs):
        root = self.generate_json()
        return json.dump(root, file_obj, **kwargs)

    def generate_json(self):
        """Header tagged with anticipated word tags?"""
        root = {FILE: self.basename}

        if self.headers:
            root[HEADER] = {}
            for column, header_dict in self.headers.items():
                root[HEADER][str(column)] = header_dict
                # root[HEADER][str(column)] = {
                #     'labels': {},
                #     'value': header,
                # }

        root[CONTENT] = {}
        for row in sorted(self.parsed.keys()):
            root[CONTENT][str(row)] = {}
            for column in sorted(self.parsed[row].keys()):
                root[CONTENT][str(row)][str(column)] = self.parsed[row][column].json_element()

        return root

    def count_labels(self):
        if not self.headers:
            raise NotImplementedError('Columns must have headers')

        counts = {}
        for row in self.parsed.keys():
            for column in self.parsed[row].keys():
                if column not in counts:
                    counts[column] = collections.defaultdict(int)
                for label, count in self.parsed[row][column].label_count().items():
                    counts[column][label] += count

        for column in self.headers.keys():
            del self.headers[column]['labels']
            self.headers[column]['labels'] = {}
            for label, count in counts[column].items():
                self.headers[column]['labels'][label] = count

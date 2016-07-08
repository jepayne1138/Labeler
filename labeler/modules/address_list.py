"""
Dict Format -> {
    HEADER: {
        int(<column>): {}
            VALUE: str,
            TAG: str
        }
    },
    CONTENT: {
        int(<row>): {
            int(<column>): {
                [
                    {
                        VALUE: str
                        TAG: str
                    }
                ]
            }
        }
    },
    FILE: str
}
"""
import collections
import os
import re
import json


# Compiled RegEx pattern
RE_WORD = re.compile("([A-Za-z0-9]+)")  # Used to split string into words

# Constants define the keys of the underlying dict structure (which )
HEADER = 'header'
CONTENT = 'content'
VALUE = 'value'
TAG = 'tag'
FILE = 'file'

# Encoding constants
ENCODING = 'utf-8'


def base_name(path):
    """Return the base filename of a path without an extension"""
    return os.path.splitext(os.path.basename(path))[0]


class AddressListEncoder(json.JSONEncoder):

    """Custom JSONEncoder that allows objects to define a to_json method"""

    def default(self, obj):
        if hasattr(obj, "to_json"):
            return super().default(obj.to_json())
        return super().default(obj)


class AddressList:

    @classmethod
    def from_excel(cls, file_obj, sheet=0, headers=True):
        import excel
        values = excel.parse_values(file_obj, sheet=sheet)
        if headers:
            header_list = values.pop(0)
        content_dict, header_dict = excel.value_dict(
            excel.sheet(file_obj, sheet=sheet),
            headers,
            excel.normalize,
            input_encoding=excel.XLSX_ENCODING,
            output_encoding=ENCODING
        )

        # Read content
        content = collections.defaultdict(dict)
        for row, row_list in enumerate(xl_sheet.get_rows()):
            if row == 1 and headers:
                continue
            for column, value in enumerate(row_list):
                clean_value = normalize(value)
                if clean_value:
                    content[row][column] = cls.split_words(clean_value)

        # Return the new AddressList instance
        return cls(base_name(file_obj.name), header_dict, content_dict)

    @staticmethod
    def split_words(string):
        """Splits a string into a list of words

        Each word is a dict with VALUE and TAG keys, and separating
        characters are just stored as a string

        Should just return the word in a list if already just one word
        """
        word_index = 0
        split = []
        for word in RE_WORD.split(string):
            if word != '':
                if RE_WORD.match(word):
                    split.append(
                        TagWord(self, word, self.row, self.column, word_index)
                    )
                    word_index += 1
                else:
                    split.append(word)
        return split

    @classmethod
    def from_json(cls, file_obj, **kwargs):
        _data = json.load(file_obj, **kwargs)
        headers = _data.get(HEADER, default={})
        content = _data.get(CONTENT, default={})
        filename = _data.get(FILE, default=base_name(file_obj.name))
        return cls(filename, headers, content)

    @classmethod
    def from_table(cls, filename, table, headers=True):
        if headers:
            header_dict = {k: v for k, v in enumerate(table.pop(0)) if v != ''}
        else:
            header_dict = {}

        # Iterate over the remaining value to create the content dict
        content_dict = collections.defaultdict(dict)
        for row, row_list in enumerate(table):
            for col, value in enumerate(row_list):
                if value != '':
                    content_dict[row][col] = cls.split_words(value)
        return cls(filename, header_dict, content_dict)


    def __init__(self, filename, headers=None, content=None):
        # Set default values
        self.headers = {} if headers is None else headers
        self.content = {} if content is None else content
        self.filename = filename

    def to_json(self, **kwargs):
        return {
            HEADER: self.headers,
            CONTENT: self.content,
            FILE: self.filename
        }

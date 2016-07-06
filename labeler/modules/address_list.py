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
import math
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
XLSX_ENCODING = "cp1252"
ENCODING = 'utf-8'


def base_name(path):
    """Return the base filename of a path without an extension"""
    return os.path.splitext(os.path.basename(path))[0]


def normalize(input_var):
    """Normalizes a variable as string, cleans and converts float to int"""
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


class AddressList:

    @classmethod
    def from_excel(cls, file_obj, sheet=0, headers=True):
        import xlrd  # Only need to import this module if parsing from Excel
        # Open the Excel workbook
        xl_book = xlrd.open_workbook(
            file_contents=file_obj,
            encoding_override=XLSX_ENCODING
        )
        xl_sheet = xl_book.sheet_by_index(sheet)

        # Read headers
        headers = {}
        if headers:
            # TODO:  Handle blank lines at the top of the file
            for column, header in enumerate(xl_sheet.row_values(0)):
                headers[column] = {VALUE: header, TAG: ''}

        # Read content
        content = collections.defaultdict(dict)
        for row, row_list in enumerate(xl_sheet.get_rows()):
            if row == 1 and headers:
                continue
            for column, value in enumerate(row_list):
                content[row][column] = normalize(value)

        # Return the new AddressList instance
        return cls(base_name(file_obj.name), headers, content)

    @classmethod
    def from_json(cls, file_obj, **kwargs):
        _data = json.load(file_obj, **kwargs)
        headers = _data.get(HEADER, default={})
        content = _data.get(CONTENT, default={})
        filename = _data.get(FILE, default=base_name(file_obj.name))
        return cls(filename, headers, content)

    def __init__(self, filename, headers=None, content=None):
        # Set default values
        headers = {} if headers is None else headers
        content = {} if content is None else content

        self._data = {
            HEADER: headers,
            CONTENT: content,
            FILE: filename
        }

    def json(self, **kwargs):
        return json.dumps(self._data, **kwargs)

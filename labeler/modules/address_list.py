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


class AddressList:

    @classmethod
    def from_excel(cls, file_obj):
        headers = {}
        content = {}
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

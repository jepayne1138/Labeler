import math
import itertools
import collections
import xlrd

# Default MS Excel encoding
XLSX_ENCODING = 'cp1252'
ENCODING = 'utf-8'


# def value_iterator(sheet):
#     """Generator returns each value in the xlrd.sheet.Sheet
#
#     Args:
#         sheet (xlrd.sheet.Sheet): Sheet object to iterate over
#         headers (bool)
#
#     Yields:
#         Tuple[int, int, str]: Row, column, and value of a cell in the sheet
#     """
#     for row, col in itertools.product(range(sheet.nrow), range(sheet.ncols)):
#        yield (row, col, sheet.cell_value(row, col))


# def value_dicts(sheet, headers, func=None, *func_args, **func_kwargs):
#     """Build nested dict of cell values with rows and columns as keys

#     Args:
#         sheet (xlrd.sheet.Sheet): Sheet object dict should be built from
#         func (optional): Callable function that is applied to the value
#             of each cell before it's added to the dict

#     Returns:
#         Dict[int, Dict[int, str]]: Nested dict of cell values with row
#             numbers as the outer keys and column numbers as the inner
#             keys, with the cell value as the inner dict value
#     """
#     # TODO:  Look more into if the parameter "func=lambda x: x" is better
#     if not func:
#         def func(value):
#             return value
#     # Create the nested dicts to return
#     header_dict = {}
#     content_dict = collections.defaultdict(dict)
#     for row, col, value in value_iterator(sheet):
#         if headers:
#             if row == 0:
#                 header_dict[col] = func(value, *func_args, **func_kwargs)
#                 continue
#             row = row - 1
#         # Apply the function to each returned value in the sheet
#         content_dict[row][col] = func(value, *func_args, **func_kwargs)
#     return (content_dict, header_dict)


def normalize(
        input_var,
        input_encoding=XLSX_ENCODING,
        output_encoding=ENCODING,
        error='ignore'):
    """Normalizes a variable as string, cleans and converts float to int"""
    var = input_var
    try:
        var = var.encode(input_encoding).decode(output_encoding, error)
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


def parse_values(file_obj, sheet=0, **kwargs):
    """Returns the normalized values of an Excel file as a 2D list"""
    xl_book = xlrd.open_workbook(file_contents=file_obj, **kwargs)
    xl_sheet = xl_book.sheet_by_index(sheet)

    return_list = []
    for row in range(xl_sheet.nrows):
        columns = xl_sheet.row_values(start_colx=0, end_colx=xl_sheet.ncols)
        return_list.append(
            [normalize(value) for value in columns]
        )
    return return_list

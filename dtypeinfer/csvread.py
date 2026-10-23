"""Read a CSV and infer its columns without holding the file in memory."""

import csv

from .infer import infer_table
from .types import Settings


def infer_csv(path, sample_size=1000, seed=0, columns=None, encoding="utf-8-sig",
              delimiter=None, **settings):
    """Infer the columns of a CSV file in one streaming pass.

    The file is read row by row and only the reservoirs are kept, so peak memory
    is set by `columns * sample_size` rather than by the size of the file.
    """
    config = Settings(sample_size=sample_size, seed=seed, **settings)
    with open(path, "r", newline="", encoding=encoding) as handle:
        if delimiter is None:
            delimiter = _sniff(handle)
            handle.seek(0)
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            return infer_table([], columns=columns, settings=config)
        return infer_table(_rows(reader), columns=columns, settings=config)


def _rows(reader):
    for row in reader:
        if None in row:
            # A row with more fields than the header parks the extras under the
            # None key. They are data, but not a column, so drop them.
            row = {key: value for key, value in row.items() if key is not None}
        yield row


def _sniff(handle):
    head = handle.read(8192)
    try:
        return csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except csv.Error:
        # One column, or a header with no separator to find: the default is fine.
        return ","

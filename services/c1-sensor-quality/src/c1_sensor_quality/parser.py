"""Reads raw sensor CSV files for C1.

Some lines in the raw file have extra quotes around the whole line. This file
removes those quotes and turns each line into a dictionary that the schema can check.
"""

import csv
from collections.abc import Iterator
from typing import TextIO

# Cell text that means "no value". We compare in lowercase.
EMPTY_CELL_VALUES = {"", "nan", "null", "none", "na", "n/a"}


def unwrap_csv_line(line: str) -> str:
    """Removes the extra quotes around a whole line.

    Example input:
        "2012-01-14 00:00:00,M001,HIGH_LOAD,...,""vibration,rpm"",align_motor,low,False"

    Example output:
        2012-01-14 00:00:00,M001,HIGH_LOAD,...,"vibration,rpm",align_motor,low,False

    A wrapped line reads as one single value. A normal line reads as many values,
    so we leave it as it is. Empty cells (,,) stay as they are.
    """
    values = next(csv.reader([line]), [])
    if len(values) == 1 and line.startswith('"'):
        return values[0]
    return line


def read_csv_rows(file_obj: TextIO) -> Iterator[tuple[int, dict[str, str | None]]]:
    """Reads a raw sensor CSV and gives back each row with its line number.

    Each row is a dictionary with the column names as keys.
    Empty cells and text like "nan" or "null" become None, so the schema sees them as missing.
    If a line has the wrong number of values, it stops with an error that names the line.
    It reads one line at a time, so big files do not fill up the memory.
    """
    reader = csv.reader(unwrap_csv_line(line) for line in file_obj)
    header = next(reader, None)
    if header is None:
        return

    # Files saved from Excel can have a hidden character before the first column name
    header[0] = header[0].removeprefix("\ufeff")

    for line_number, values in enumerate(reader, start=2):
        if len(values) == 0:
            continue  # Skip blank lines

        if len(values) != len(header):
            raise ValueError(
                f"Line {line_number}: expected {len(header)} values but found {len(values)}."
            )

        row: dict[str, str | None] = {}
        for column_name, value in zip(header, values):
            row[column_name] = None if value.strip().lower() in EMPTY_CELL_VALUES else value
        yield line_number, row

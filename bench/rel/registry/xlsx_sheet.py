"""Transcribe one sheet of an .xlsx supplement as it stands, for the event registry (package W15). Standard library only.

usage: python xlsx_sheet.py BOOK.xlsx SHEET_NUMBER > out.tsv

Writes one line per spreadsheet row: the row number, then for every cell `column=value` with the cell's style id in
brackets (`C=2690[56]`), so colour-coded classes stay recoverable without interpreting them. Shared strings are
resolved; nothing is computed, rounded or dropped. The first comment line gives the book's sha256 and the sheet. Used
on the servers on files a CI run stored with checksums; registry rows cite the output file and row numbers.
"""
from __future__ import annotations

import hashlib
import re
import sys
import zipfile


def rows(book, sheet):
    z = zipfile.ZipFile(book)
    strings = []
    if "xl/sharedStrings.xml" in z.namelist():
        ss = z.read("xl/sharedStrings.xml").decode("utf-8")
        strings = [re.sub(r"<[^>]+>", "", m) for m in re.findall(r"<si>(.*?)</si>", ss, re.S)]
    data = z.read("xl/worksheets/sheet%d.xml" % sheet).decode("utf-8")
    for r in re.findall(r"<row ([^>]*)>(.*?)</row>", data, re.S):
        num = re.search(r'r="(\d+)"', r[0]).group(1)
        cells = []
        for c in re.finditer(r'<c r="([A-Z]+)\d+"([^>]*?)(/>|>(.*?)</c>)', r[1], re.S):
            col, attrs, inner = c.group(1), c.group(2), c.group(4) or ""
            v = re.search(r"<v>(.*?)</v>", inner, re.S)
            t = re.search(r't="(\w+)"', attrs)
            s = re.search(r's="(\d+)"', attrs)
            val = v.group(1) if v else re.sub(r"<[^>]+>", "", inner)
            if t and t.group(1) == "s" and v:
                val = strings[int(val)]
            val = val.replace("\t", " ").replace("\n", " ").replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
            if val or s:
                cells.append("%s=%s[%s]" % (col, val, s.group(1) if s else ""))
        yield num, cells


def main(argv):
    book, sheet = argv[1], int(argv[2])
    with open(book, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    print("# %s sheet %d sha256 %s; cells as column=value[style id]" % (book.rsplit("/", 1)[-1], sheet, sha))
    for num, cells in rows(book, sheet):
        print("\t".join([num] + cells))


if __name__ == "__main__":
    main(sys.argv)

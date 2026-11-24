"""Text of a PDF as it stands, page by page, for the event registry (package W15) and any other PDF supplement.
Standard library only.

usage: python -I pdf_text.py FILE.pdf [--sha256 HEX] > out.txt

Reads the file's objects (plain and inside object streams), decodes the content streams (FlateDecode, LZWDecode,
ASCIIHexDecode, ASCII85Decode, RunLengthDecode, with the PNG and TIFF predictors) and takes the text of the
text-showing operators (Tj, TJ, ', "), also inside form XObjects, with the font kept in the graphics state (q and Q
save and restore it; a form starts with its caller's font). Characters come only from what the file states: a font's
ToUnicode map; a simple font's named encoding (WinAnsi, MacRoman as the PDF standard defines them, Standard) and its
Differences, with glyph names read by the Adobe glyph naming rules for the names this module knows; the Standard
encoding for the non-symbolic standard 14 fonts used as such (not embedded subsets). Anything else (a composite font
without ToUnicode, a font whose own built-in encoding the file does not state, an unknown glyph name) is not guessed:
its codes come out as U+FFFD and count as unreadable.

Limits, so that no file can hold the select lock for long: at most STREAM_MAX bytes decoded per stream and DOC_MAX per
document (Flate in 64 KB pieces; a broken piece is fed byte by byte to find where it breaks, and a page using a broken
stream is unreadable with that place named, since corrupt data may decode to wrong text before zlib notices; data cut
short keep their text with a note);
per page at most PAGE_TOKENS tokens, PAGE_DO form calls and PAGE_CHARS characters, per document DOC_TOKENS tokens;
decoded streams are cached by object number. A page over a limit is unreadable, with the limit named.

A page is readable when it has text and
- at least MIN_RATIO of its characters (whitespace aside) are readable (no U+FFFD, control, private-use, unassigned or
  surrogate code point);
- with MIN_WORDS or more letter-only words of four letters or more, most of their letters Latin (otherwise the page is
  unverified: the word measures cannot check another script, and it gives no text either): at least MIN_WORDLIKE of
  them contain a vowel (a e i o u y, accented ones counted, other non-ASCII letters not), and the vowels (a e i o u,
  accents removed) make VOWELS of their Latin letters;
- with MIN_WORDS_PAIRS or more such words: at least MIN_PAIRS of their letter pairs (accents removed) are among
  the 100 commonest pairs of English (the supplements read here are in English).
Sequence-like tokens (nucleotide letters only, or 15 or more capitals) are not words for these measures.
The last two guard against a valid-looking but shifted character map. Only readable pages give text; every other page
gives its reason and no text, so nothing unreadable enters the registry. An encrypted file, or one whose structure
cannot be read, is unreadable as a whole. Spaces and line breaks come from the text positioning operators and are
layout, not content.

Output: a header line with the file's sha256, then per page a line '## page N: status (...)' and, for readable pages
only, the text. Every string taken from the file and written outside the page text (reasons, notes, font names) has
its control characters escaped (\\xNN); in the page text, control and line-separator characters other than newline
and tab are escaped the same way, and a line starting with '#' or '\\' is written with a leading '\\'. With --sha256 a
file that is not the stored one is refused (exit 2). Used on the servers on files a CI run stored with checksums;
registry rows cite the output file, page and line.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import unicodedata
import zlib
from collections import deque

VERSION = "0.3"
MIN_RATIO = 0.95
MIN_WORDS = 20
MIN_WORDLIKE = 0.90
VOWELS = (0.30, 0.55)          # English prose about 0.38; Latin binomials run high (Alloteropsis semialata 0.50)
MIN_WORDS_PAIRS = 20           # measured: real text 0.72 or more over any 20 to 49 words, letter-shifted text 0.36 at most
MIN_PAIRS = 0.45
SEQUENCE = re.compile(r"[ACGTUNacgtun]+|[A-Z]{15,}")   # nucleotide or protein sequence: no word, left out of the word measures
REPLACEMENT = "�"
MAX_DEPTH = 8                  # nesting of form XObjects and of indirect references
STREAM_MAX = 64 * 2 ** 20
DOC_MAX = 256 * 2 ** 20
CHUNK = 64 * 2 ** 10
PAGE_TOKENS = 5_000_000
PAGE_DO = 2_000
PAGE_CHARS = 2_000_000
DOC_TOKENS = 50_000_000
MAX_OPERANDS = 10_000
COMMON_PAIRS = set(
    "th he in er an re on at en nd ti es or te of ed is it al ar st to nt ng se ha as ou io le ve co me de hi ri ro ic ne "
    "ea ra ce li ch ll be ma si om ur ca el ta la ns di fo ho pe ec pr no ct us ac ot il tr ly nc et ut ss so rs un lo wa "
    "ge ie wh ee wi em ad ol rt po we na ul ni ts mo ow pa im mi ai sh".split())


class Unreadable(Exception):
    """The file as a whole cannot be read (encrypted, no page tree, broken structure, over the document limits)."""


class Budget(Exception):
    """A limit of decoding or interpretation was reached."""


class Broken(ValueError):
    """Compressed data that fails to decode: what came before the failure may already be wrong, so none of it is used."""


class Name(str):
    pass


class Op(str):
    pass


class Ref(tuple):
    pass


def safe(text, keep=""):
    """text with its control, format and line-separator characters escaped (those in keep left as they are)."""
    out = []
    for c in str(text):
        if c not in keep and unicodedata.category(c) in ("Cc", "Cf", "Zl", "Zp"):
            out.append("\\x%02x" % ord(c) if ord(c) < 256 else "\\u%04x" % ord(c))
        else:
            out.append(c)
    return "".join(out)


def describe(x):
    """A PDF value in words, never an object's repr."""
    if isinstance(x, Name):
        return "/" + safe(x)
    if isinstance(x, Stream):
        return "an embedded stream (object %s)" % x.num
    if x is None:
        return "nothing"
    return {dict: "a dictionary", list: "an array", bytes: "a string", bool: "a boolean"}.get(type(x), "a number" if isinstance(x, (int, float)) else "a value")


class Stream:
    def __init__(self, doc, d, start, end, num, closed):
        self.doc, self.dict, self.start, self.end, self.num, self.closed = doc, d, start, end, num, closed

    def raw(self):
        n = self.doc.resolve(self.dict.get("Length"))
        data = self.doc.data
        if isinstance(n, int) and 0 <= n and re.match(rb"\s*endstream", data[self.start + n:self.start + n + 20]):
            return data[self.start:self.start + n], None
        note = None if self.closed else "stream %s has no endstream: read to the end of the file" % self.num
        return data[self.start:self.end], note

    def decoded(self):
        """(bytes, note) of the stream through its filters, within STREAM_MAX and what is left of DOC_MAX; cached."""
        doc = self.doc
        if self.num in doc.cache:
            got = doc.cache[self.num]
            if isinstance(got, Exception):
                raise got
            return got
        filters = doc.resolve(self.dict.get("Filter"))
        parms = doc.resolve(self.dict.get("DecodeParms"))
        filters = filters if isinstance(filters, list) else ([filters] if filters else [])
        if isinstance(parms, list):
            parms = (parms + [None] * len(filters))[:len(filters)]
        elif parms is None or len(filters) == 1:
            parms = [parms] * len(filters)
        else:
            raise ValueError("DecodeParms does not match the %d filters of stream %s" % (len(filters), self.num))
        data, note = self.raw()
        notes = [note] if note else []
        limit = min(STREAM_MAX, DOC_MAX - doc.decoded)
        for f, p in zip(filters, parms):
            try:
                data, n = decode(doc.resolve(f), data, doc.resolve(p), limit)
            except Broken as err:
                doc.cache[self.num] = Broken("stream %s: %s" % (self.num, err))
                raise doc.cache[self.num] from None
            if n:
                notes.append("stream %s: %s" % (self.num, n))
        doc.decoded += len(data)
        if doc.decoded > DOC_MAX:
            raise Budget("more than %d MB decoded in the document" % (DOC_MAX >> 20))
        doc.cache[self.num] = (data, "; ".join(notes) or None)
        return doc.cache[self.num]


# ---------------------------------------------------------------- lexer and parser

_DELIM = rb"\x00\t\n\x0c\r ()<>\[\]{}/%"
TOKEN = re.compile(rb"(?P<ws>(?:[\x00\t\n\x0c\r ]+|%[^\r\n]*)+)"
                   rb"|(?P<num>[+-]?(?:\d+\.?\d*|\.\d+))(?=[" + _DELIM + rb"]|$)"
                   rb"|(?P<name>/[^" + _DELIM + rb"]*)"
                   rb"|(?P<dopen><<)|(?P<dclose>>>)"
                   rb"|(?P<hex><[0-9A-Fa-f\x00\t\n\x0c\r ]*>)"
                   rb"|(?P<lit>\()"
                   rb"|(?P<aopen>\[)|(?P<aclose>\])|(?P<brace>[{}])"
                   rb"|(?P<kw>[^" + _DELIM + rb"]+)")
_LIT_SPECIAL = re.compile(rb"[()\\]")
_ESC = {ord("n"): 10, ord("r"): 13, ord("t"): 9, ord("b"): 8, ord("f"): 12}


def _literal(data, i):
    """The literal string starting after the '(' at i - 1: (bytes, position after the closing ')')."""
    out, depth = bytearray(), 1
    while True:
        m = _LIT_SPECIAL.search(data, i)
        if not m:
            out += data[i:]
            return bytes(out), len(data)
        out += data[i:m.start()]
        i = m.start()
        c = data[i]
        if c == 0x5C:
            e = data[i + 1] if i + 1 < len(data) else None
            if e is None:
                return bytes(out), len(data)
            if e in _ESC:
                out.append(_ESC[e])
                i += 2
            elif 0x30 <= e <= 0x37:
                j, v = i + 1, 0
                while j < len(data) and j < i + 4 and 0x30 <= data[j] <= 0x37:
                    v = v * 8 + data[j] - 0x30
                    j += 1
                out.append(v & 0xFF)
                i = j
            elif e == 0x0D:
                i += 3 if data[i + 2:i + 3] == b"\n" else 2
            elif e == 0x0A:
                i += 2
            else:
                out.append(e)
                i += 2
        elif c == 0x28:
            depth += 1
            out.append(c)
            i += 1
        else:
            depth -= 1
            if depth == 0:
                return bytes(out), i + 1
            out.append(c)
            i += 1


def _name(raw):
    return Name(re.sub(rb"#([0-9A-Fa-f]{2})", lambda m: bytes([int(m.group(1), 16)]), raw[1:]).decode("latin-1"))


def tokens(data, i=0):
    """Tokens of PDF syntax from position i: (token, position after it). Strings come as bytes, names as Name,
    numbers as int or float, everything else as Op."""
    n = len(data)
    while i < n:
        m = TOKEN.match(data, i)
        if not m:                                      # a stray delimiter (')', '>'): skip it
            i += 1
            continue
        kind, i = m.lastgroup, m.end()
        if kind == "ws":
            continue
        if kind == "num":
            t = m.group(0)
            yield (float(t) if b"." in t else int(t)), i
        elif kind == "name":
            yield _name(m.group(0)), i
        elif kind == "lit":
            s, i = _literal(data, i)
            yield s, i
        elif kind == "hex":
            h = re.sub(rb"[^0-9A-Fa-f]", b"", m.group(0))
            yield bytes.fromhex((h + b"0" if len(h) % 2 else h).decode()), i
        else:
            yield Op(m.group(0).decode("latin-1")), i


class Parser:
    """Objects of PDF syntax from position i; `pos` is the position after the last token taken."""

    def __init__(self, data, i=0):
        self.data, self.it, self.ahead, self.pos = data, tokens(data, i), deque(), i

    def restart(self, i):
        self.it, self.ahead, self.pos = tokens(self.data, i), deque(), i

    def _fill(self, k):
        while len(self.ahead) <= k:
            try:
                self.ahead.append(next(self.it))
            except StopIteration:
                return False
        return True

    def peek(self, k=0):
        return self.ahead[k][0] if self._fill(k) else None

    def next(self):
        if not self._fill(0):
            raise EOFError
        t, self.pos = self.ahead.popleft()
        return t

    def object(self, t=None):
        t = self.next() if t is None else t
        if isinstance(t, Op):
            if t == "<<":
                d = {}
                while True:
                    k = self.next()
                    if k == ">>":
                        return d
                    v = self.object()
                    if isinstance(k, Name):
                        d[str(k)] = v
            if t == "[":
                out = []
                while True:
                    k = self.next()
                    if k == "]":
                        return out
                    out.append(self.object(k))
            return {"true": True, "false": False, "null": None}.get(t, t)
        if isinstance(t, int) and not isinstance(t, bool):
            g, r = self.peek(0), self.peek(1)
            if isinstance(g, int) and r == "R":
                self.next()
                self.next()
                return Ref((t, g))
        return t


# ---------------------------------------------------------------- filters

def _over(n, limit):
    if n > limit:
        raise Budget("a stream decodes to more than %d MB" % max(1, limit >> 20))


def _feed(d, data, out, limit):
    buf = data
    while True:
        out += d.decompress(buf, limit + 1 - len(out))
        _over(len(out), limit)
        buf = d.unconsumed_tail
        if not buf or d.eof:
            return


def _inflate(data, limit):
    """Flate data in CHUNK pieces, at most limit bytes out: (bytes, note), the note saying when the data stop before their
    end. A piece that fails is fed again byte by byte from the state before it to find the byte where it breaks, which
    Broken names; corrupt data can decode to wrong bytes before zlib notices, so a broken stream gives no text."""
    d, out = zlib.decompressobj(), bytearray()
    for pos in range(0, len(data), CHUNK):
        piece = data[pos:pos + CHUNK]
        saved, n0 = d.copy(), len(out)
        try:
            _feed(d, piece, out, limit)
        except zlib.error:
            del out[n0:]
            d = saved
            for k in range(len(piece)):
                try:
                    _feed(d, piece[k:k + 1], out, limit)
                except zlib.error:
                    raise Broken("compressed data broken at byte %d of %d" % (pos + k, len(data))) from None
                if d.eof:
                    break
        if d.eof:
            return bytes(out), None
    return bytes(out), ("compressed data cut short (%d bytes, no end of data)" % len(data) if data else None)


def _lzw(data, early=1, limit=STREAM_MAX):
    out, table, prev = bytearray(), [bytes([i]) for i in range(256)] + [None, None], None
    buf, nbits, width = 0, 0, 9
    for byte in data:
        buf = (buf << 8) | byte
        nbits += 8
        while nbits >= width:
            nbits -= width
            code = (buf >> nbits) & ((1 << width) - 1)
            buf &= (1 << nbits) - 1
            if code == 256:
                table, prev, width = table[:258], None, 9
                continue
            if code == 257:
                return bytes(out), None
            if prev is None:
                if code >= len(table) or table[code] is None:
                    raise ValueError("LZW data broken")
                entry = table[code]
            elif code < len(table):
                entry = table[code]
                if len(table) < 4096:
                    table.append(prev + entry[:1])
            elif code == len(table) and len(table) < 4096:
                entry = prev + prev[:1]
                table.append(entry)
            else:
                raise ValueError("LZW data broken")
            out += entry
            _over(len(out), limit)
            prev = entry
            if len(table) + early >= (1 << width) and width < 12:
                width += 1
    return bytes(out), "LZW data without an end code"


def _a85(data):
    data = re.sub(rb"\s", b"", data)
    if data.startswith(b"<~"):
        data = data[2:]
    data = data.split(b"~>")[0]
    out, group = bytearray(), []
    for c in data:
        if c == ord("z") and not group:
            out += b"\0\0\0\0"
            continue
        if not 33 <= c <= 117:
            raise ValueError("ASCII85 data broken")
        group.append(c - 33)
        if len(group) == 5:
            v = 0
            for x in group:
                v = v * 85 + x
            if v >= 1 << 32:
                raise ValueError("ASCII85 data broken")
            out += v.to_bytes(4, "big")
            group = []
    if group:
        k = len(group)
        v = 0
        for x in group + [84] * (5 - k):
            v = v * 85 + x
        out += (v % (1 << 32)).to_bytes(4, "big")[:k - 1]
    return bytes(out)


def _runlength(data, limit):
    out, i = bytearray(), 0
    while i < len(data):
        n = data[i]
        if n == 128:
            break
        if n < 128:
            out += data[i + 1:i + 2 + n]
            i += n + 2
        else:
            out += data[i + 1:i + 2] * (257 - n)
            i += 2
        _over(len(out), limit)
    return bytes(out)


def _predict(data, parms):
    pred = (parms or {}).get("Predictor", 1)
    if pred == 1:
        return data
    colors, bpc = parms.get("Colors", 1), parms.get("BitsPerComponent", 8)
    cols = parms.get("Columns", 1)
    if not all(isinstance(x, int) and x > 0 for x in (colors, bpc, cols)):
        raise ValueError("predictor parameters are not read")
    bpp = max(1, colors * bpc // 8)
    width = (cols * colors * bpc + 7) // 8
    if pred == 2:                                       # TIFF: each byte adds the one bpp to its left, 8 bits only
        if bpc != 8:
            raise ValueError("TIFF predictor with %d bits per component is not read" % bpc)
        out = bytearray(data)
        for k in range(0, len(out), width):
            for j in range(k + bpp, min(k + width, len(out))):
                out[j] = (out[j] + out[j - bpp]) & 0xFF
        return bytes(out)
    if not isinstance(pred, int) or not 10 <= pred <= 15:
        raise ValueError("predictor %s is not read" % (pred if isinstance(pred, int) else describe(pred)))
    out, prev = bytearray(), bytearray(width)
    for k in range(0, len(data), width + 1):
        ft, row = data[k], bytearray(data[k + 1:k + 1 + width])
        if ft > 4:
            raise ValueError("PNG predictor row type %d is not read" % ft)
        for j in range(len(row)):
            a = row[j - bpp] if j >= bpp else 0
            b, c = prev[j], (prev[j - bpp] if j >= bpp else 0)
            if ft == 1:
                row[j] = (row[j] + a) & 0xFF
            elif ft == 2:
                row[j] = (row[j] + b) & 0xFF
            elif ft == 3:
                row[j] = (row[j] + (a + b) // 2) & 0xFF
            elif ft == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                row[j] = (row[j] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 0xFF
        out += row
        prev = row + bytearray(width - len(row))
    return bytes(out)


def decode(name, data, parms=None, limit=STREAM_MAX):
    """(bytes, note) of data through one filter, at most limit bytes."""
    parms = parms if isinstance(parms, dict) else None
    if name in ("FlateDecode", "Fl"):
        out, note = _inflate(data, limit)
        return _predict(out, parms), note
    if name in ("LZWDecode", "LZW"):
        out, note = _lzw(data, (parms or {}).get("EarlyChange", 1), limit)
        return _predict(out, parms), note
    if name in ("ASCIIHexDecode", "AHx"):
        h = re.sub(rb"[^0-9A-Fa-f]", b"", data.split(b">")[0])
        out = bytes.fromhex((h + b"0" if len(h) % 2 else h).decode())
    elif name in ("ASCII85Decode", "A85"):
        out = _a85(data)
    elif name in ("RunLengthDecode", "RL"):
        out = _runlength(data, limit)
    else:
        raise ValueError("filter %s is not read" % describe(name))
    _over(len(out), limit)
    return out, None


# ---------------------------------------------------------------- encodings and glyph names

_STD_HIGH = {
    0xA1: "exclamdown", 0xA2: "cent", 0xA3: "sterling", 0xA4: "fraction", 0xA5: "yen", 0xA6: "florin", 0xA7: "section",
    0xA8: "currency", 0xA9: "quotesingle", 0xAA: "quotedblleft", 0xAB: "guillemotleft", 0xAC: "guilsinglleft",
    0xAD: "guilsinglright", 0xAE: "fi", 0xAF: "fl", 0xB1: "endash", 0xB2: "dagger", 0xB3: "daggerdbl",
    0xB4: "periodcentered", 0xB6: "paragraph", 0xB7: "bullet", 0xB8: "quotesinglbase", 0xB9: "quotedblbase",
    0xBA: "quotedblright", 0xBB: "guillemotright", 0xBC: "ellipsis", 0xBD: "perthousand", 0xBF: "questiondown",
    0xC1: "grave", 0xC2: "acute", 0xC3: "circumflex", 0xC4: "tilde", 0xC5: "macron", 0xC6: "breve", 0xC7: "dotaccent",
    0xC8: "dieresis", 0xCA: "ring", 0xCB: "cedilla", 0xCD: "hungarumlaut", 0xCE: "ogonek", 0xCF: "caron", 0xD0: "emdash",
    0xE1: "AE", 0xE3: "ordfeminine", 0xE8: "Lslash", 0xE9: "Oslash", 0xEA: "OE", 0xEB: "ordmasculine", 0xF1: "ae",
    0xF5: "dotlessi", 0xF8: "lslash", 0xF9: "oslash", 0xFA: "oe", 0xFB: "germandbls"}
_NAMED = {
    "space": " ", "exclam": "!", "quotedbl": '"', "numbersign": "#", "dollar": "$", "percent": "%", "ampersand": "&",
    "quotesingle": "'", "quoteright": "’", "quoteleft": "‘", "parenleft": "(", "parenright": ")",
    "asterisk": "*", "plus": "+", "comma": ",", "hyphen": "-", "period": ".", "slash": "/", "colon": ":",
    "semicolon": ";", "less": "<", "equal": "=", "greater": ">", "question": "?", "at": "@", "bracketleft": "[",
    "backslash": "\\", "bracketright": "]", "asciicircum": "^", "underscore": "_", "grave": "`", "braceleft": "{",
    "bar": "|", "braceright": "}", "asciitilde": "~", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "exclamdown": "¡", "cent": "¢",
    "sterling": "£", "fraction": "⁄", "yen": "¥", "florin": "ƒ", "section": "§",
    "currency": "¤", "quotedblleft": "“", "guillemotleft": "«", "guilsinglleft": "‹",
    "guilsinglright": "›", "fi": "fi", "fl": "fl", "ff": "ff", "ffi": "ffi", "ffl": "ffl", "endash": "–",
    "dagger": "†", "daggerdbl": "‡", "periodcentered": "·", "paragraph": "¶", "bullet": "•",
    "quotesinglbase": "‚", "quotedblbase": "„", "quotedblright": "”", "guillemotright": "»",
    "ellipsis": "…", "perthousand": "‰", "questiondown": "¿", "acute": "´", "circumflex": "ˆ",
    "tilde": "˜", "macron": "¯", "breve": "˘", "dotaccent": "˙", "dieresis": "¨",
    "ring": "˚", "cedilla": "¸", "hungarumlaut": "˝", "ogonek": "˛", "caron": "ˇ",
    "emdash": "—", "AE": "Æ", "ordfeminine": "ª", "Lslash": "Ł", "Oslash": "Ø", "OE": "Œ",
    "ordmasculine": "º", "ae": "æ", "dotlessi": "ı", "lslash": "ł", "oslash": "ø",
    "oe": "œ", "germandbls": "ß", "minus": "−", "multiply": "×", "divide": "÷",
    "plusminus": "±", "degree": "°", "mu": "µ", "copyright": "©", "registered": "®",
    "trademark": "™", "logicalnot": "¬", "brokenbar": "¦", "onehalf": "½", "onequarter": "¼",
    "threequarters": "¾", "onesuperior": "¹", "twosuperior": "²", "threesuperior": "³",
    "Euro": "€", "nbspace": " ", "sfthyphen": "-", "infinity": "∞", "lessequal": "≤",
    "greaterequal": "≥", "notequal": "≠", "approxequal": "≈", "arrowright": "→",
    "arrowleft": "←", "partialdiff": "∂", "radical": "√", "summation": "∑", "product": "∏",
    "integral": "∫", "Eth": "Ð", "eth": "ð", "Thorn": "Þ", "thorn": "þ"}
_WITH = {"ACUTE": "acute", "GRAVE": "grave", "CIRCUMFLEX": "circumflex", "DIAERESIS": "dieresis", "TILDE": "tilde",
         "RING ABOVE": "ring", "CEDILLA": "cedilla", "CARON": "caron", "MACRON": "macron", "BREVE": "breve",
         "OGONEK": "ogonek", "DOT ABOVE": "dotaccent", "DOUBLE ACUTE": "hungarumlaut", "STROKE": "slash"}


def _glyph_table():
    g = {c: c for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    g.update(_NAMED)
    for cp in list(range(0xC0, 0x250)) + list(range(0x391, 0x3CA)):
        ch = chr(cp)
        nm = unicodedata.name(ch, "")
        m = re.fullmatch(r"LATIN (SMALL|CAPITAL) LETTER ([A-Z]) WITH (.+)", nm)
        if m and m.group(3) in _WITH:
            base = m.group(2).lower() if m.group(1) == "SMALL" else m.group(2)
            g.setdefault(base + _WITH[m.group(3)], ch)
        m = re.fullmatch(r"GREEK (SMALL|CAPITAL) LETTER ([A-Z]+)", nm)
        if m:
            w = m.group(2).lower()
            g.setdefault(w if m.group(1) == "SMALL" else w.capitalize(), ch)
    return g


GLYPHS = _glyph_table()


def glyph(name):
    """The characters of a glyph name by the Adobe glyph naming rules (suffix after '.' dropped, '_' joins ligature
    components, uniXXXX and uXXXX[XX]); None for a name this module does not know."""
    name = str(name).split(".")[0]
    if not name:
        return None
    parts = name.split("_")
    out = []
    for p in parts:
        if p in GLYPHS:
            out.append(GLYPHS[p])
        elif re.fullmatch(r"uni(?:[0-9A-F]{4})+", p):
            cps = [int(p[k:k + 4], 16) for k in range(3, len(p), 4)]
            if any(0xD800 <= c <= 0xDFFF for c in cps):
                return None
            out.append("".join(map(chr, cps)))
        elif re.fullmatch(r"u[0-9A-F]{4,6}", p) and int(p[1:], 16) <= 0x10FFFF and not 0xD800 <= int(p[1:], 16) <= 0xDFFF:
            out.append(chr(int(p[1:], 16)))
        else:
            return None
    return "".join(out)


MACROMAN_UNDEFINED = {0xAD, 0xB0, 0xB2, 0xB3, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xBD, 0xC3, 0xC5, 0xC6, 0xD7, 0xF0}
# the mathematical signs and the Apple logo of Mac OS Roman, which the PDF standard's MacRomanEncoding does not define


def _encoding(name):
    if name == "WinAnsiEncoding":
        out = {}
        for c in range(0x20, 0x100):
            try:
                out[c] = bytes([c]).decode("cp1252")
            except UnicodeDecodeError:
                continue
        out.pop(0x7F, None)
        return out
    if name == "MacRomanEncoding":
        out = {c: bytes([c]).decode("mac_roman") for c in range(0x20, 0x100) if c != 0x7F and c not in MACROMAN_UNDEFINED}
        out[0xDB] = "¤"
        return out
    if name == "StandardEncoding":
        out = {c: chr(c) for c in range(0x20, 0x7F)}
        out[0x27], out[0x60] = "’", "‘"
        out.update({c: _NAMED[n] for c, n in _STD_HIGH.items()})
        return out
    return None


STANDARD14_TEXT = {"Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic", "Helvetica", "Helvetica-Bold",
                   "Helvetica-Oblique", "Helvetica-BoldOblique", "Courier", "Courier-Bold", "Courier-Oblique",
                   "Courier-BoldOblique"}
UNICODE_CMAPS = re.compile(r"Uni(GB|CNS|JIS|KS)-(UCS2|UTF16)-[HV]")


# ---------------------------------------------------------------- fonts

def _utf16(b):
    if isinstance(b, bytes):
        return b.decode("utf-16-be", errors="replace")
    if isinstance(b, Name):
        return glyph(b)
    return None


def _parse_cmap(data):
    """A ToUnicode CMap: (codespace ranges [(length, lo, hi)], {code bytes: text}, ranges [(length, lo, hi, dst)]),
    dst the first destination (bytes, incremented per code) or a list of texts. Ranges stay ranges."""
    space, single, ranges = [], {}, []
    toks = [t for t, _ in tokens(data)]
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "begincodespacerange":
            i += 1
            while i + 1 < len(toks) and toks[i] != "endcodespacerange":
                lo, hi = toks[i], toks[i + 1]
                if isinstance(lo, bytes) and isinstance(hi, bytes) and len(lo) == len(hi) and lo:
                    space.append((len(lo), lo, hi))
                i += 2
        elif t == "beginbfchar":
            i += 1
            while i + 1 < len(toks) and toks[i] != "endbfchar":
                src, dst = toks[i], toks[i + 1]
                if isinstance(src, bytes):
                    u = _utf16(dst)
                    if u is not None:
                        single[src] = u
                i += 2
        elif t == "beginbfrange":
            i += 1
            while i + 2 < len(toks) and toks[i] != "endbfrange":
                lo, hi = toks[i], toks[i + 1]
                if toks[i + 2] == "[":
                    j, dsts = i + 3, []
                    while j < len(toks) and toks[j] != "]":
                        dsts.append(_utf16(toks[j]))
                        j += 1
                    nxt = j + 1
                else:
                    dsts, nxt = toks[i + 2], i + 3
                if isinstance(lo, bytes) and isinstance(hi, bytes) and len(lo) == len(hi) and lo:
                    a, b = int.from_bytes(lo, "big"), int.from_bytes(hi, "big")
                    if a <= b and (isinstance(dsts, list) or isinstance(dsts, bytes) and dsts):
                        ranges.append((len(lo), a, b, dsts))
                i = nxt
        else:
            i += 1
    return space, single, ranges


class Font:
    def __init__(self, doc, d, key="?"):
        d = d if isinstance(d, dict) else {}
        self.key = key
        self.subtype = str(doc.resolve(d.get("Subtype")) or "?")
        raw = str(doc.resolve(d.get("BaseFont")) or "?")
        self.subset = bool(re.match(r"^[A-Z]{6}\+", raw))
        self.basefont = raw[7:] if self.subset else raw
        self.space, self.single, self.ranges, self.byte, self.memo = [], {}, [], {}, {}
        self.how, self.problem, self.unicode_cmap = None, None, False
        tu = doc.resolve(d.get("ToUnicode"))
        if isinstance(tu, Stream):
            try:
                self.space, self.single, self.ranges = _parse_cmap(tu.decoded()[0])
                self.how = "ToUnicode"
            except Budget:
                raise
            except Exception as err:  # noqa: BLE001 - a broken map is reported, never guessed around
                self.problem = "ToUnicode map not read (%s)" % type(err).__name__
        enc = doc.resolve(d.get("Encoding"))
        if self.subtype == "Type0":
            if not self.space:
                self.space = [(2, b"\x00\x00", b"\xff\xff")]
            if self.how is None and self.problem is None:
                if isinstance(enc, Name) and UNICODE_CMAPS.fullmatch(enc):
                    self.how, self.unicode_cmap = "Unicode CMap %s" % enc, True
                else:
                    self.problem = "composite font without ToUnicode (encoding %s: glyph codes, not characters)" % describe(enc)
            return
        self.space = self.space or [(1, b"\x00", b"\xff")]
        if self.how == "ToUnicode":
            return
        desc = doc.resolve(d.get("FontDescriptor"))
        flags = doc.resolve(desc.get("Flags")) if isinstance(desc, dict) else None
        symbolic = isinstance(flags, int) and bool(flags & 4) and not flags & 32
        base, diffs = None, {}
        if isinstance(enc, Name):
            base = _encoding(enc)
            if base is None:
                self.problem = "encoding %s is not read" % describe(enc)
        elif isinstance(enc, dict):
            be = doc.resolve(enc.get("BaseEncoding"))
            base = _encoding(be) if be else None
            code = None
            for x in doc.resolve(enc.get("Differences")) or []:
                x = doc.resolve(x)
                if isinstance(x, int):
                    code = x
                elif isinstance(x, Name) and code is not None:
                    diffs[code] = glyph(x)
                    code += 1
        elif enc is not None:
            self.problem = "encoding %s is not read" % describe(enc)
        if base is None and not self.problem and self.basefont in STANDARD14_TEXT and not self.subset and not symbolic:
            base = _encoding("StandardEncoding")
        if base is not None and symbolic:
            base = None                                # a symbolic font's codes are its own: only named glyphs count
        self.byte = dict(base or {})
        for c, u in diffs.items():
            if u is None:
                self.byte.pop(c, None)
            else:
                self.byte[c] = u
        if base is not None:
            self.how = "encoding"
        elif diffs and any(u is not None for u in diffs.values()):
            self.how = "Differences only"
        elif not self.problem:
            self.problem = ("symbolic font without ToUnicode" if symbolic else
                            "no ToUnicode and no stated encoding (the font program's own encoding is not read)")

    def _lookup(self, code):
        if code in self.memo:
            return self.memo[code]
        u = self.single.get(code)
        if u is None and self.ranges:
            v = int.from_bytes(code, "big")
            for n, a, b, dst in self.ranges:
                if n == len(code) and a <= v <= b:
                    if isinstance(dst, list):
                        u = dst[v - a] if v - a < len(dst) else None
                    else:
                        w = int.from_bytes(dst, "big") + v - a
                        u = _utf16(w.to_bytes(len(dst), "big")) if w < 1 << (8 * len(dst)) else None
                    break
        if u is None and self.byte and len(code) == 1:
            u = self.byte.get(code[0])
        if u is None and self.unicode_cmap and len(code) == 2:
            u = code.decode("utf-16-be", errors="replace")
        self.memo[code] = u
        return u

    def decode(self, s):
        """(text, unmapped codes) of a string shown in this font."""
        out, bad, i = [], 0, 0
        while i < len(s):
            n = None
            for length, lo, hi in self.space:
                code = s[i:i + length]
                if len(code) == length and all(lo[k] <= code[k] <= hi[k] for k in range(length)):
                    n = length
                    break
            n = n or self.space[0][0]
            code = s[i:i + n]
            i += n
            u = self._lookup(code)
            if u is None:
                bad += 1
                out.append(REPLACEMENT)
            else:
                out.append(u)
        return "".join(out), bad


# ---------------------------------------------------------------- document

OBJ_RE = re.compile(rb"(?<![0-9])(\d+)[\x00\t\n\x0c\r ]+(\d+)[\x00\t\n\x0c\r ]+obj(?![^" + _DELIM + rb"])")


class Document:
    def __init__(self, data):
        if b"%PDF-" not in data[:1024]:
            raise Unreadable("not a PDF (no %PDF- header)")
        self.data, self.objs, self.cache, self.decoded, self.tokens = data, {}, {}, 0, 0
        skip_until = 0
        streams = []
        for m in OBJ_RE.finditer(data):
            if m.start() < skip_until:
                continue
            num = int(m.group(1))
            try:
                p = Parser(data, m.end())
                val = p.object()
                pos = p.pos
            except (EOFError, ValueError, IndexError, RecursionError):
                continue
            if isinstance(val, dict):
                sm = re.match(rb"[\x00\t\n\x0c\r ]*stream(\r\n|\n|\r)", data[pos:pos + 40])
                if sm:
                    start = pos + sm.end()
                    n = val.get("Length")
                    end = data.find(b"endstream", start)
                    closed = end >= 0
                    end = end if closed else len(data)
                    if isinstance(n, int) and re.match(rb"\s*endstream", data[start + n:start + n + 20]):
                        skip_until = start + n
                    else:
                        skip_until = end
                    raw_end = end
                    while raw_end > start and data[raw_end - 1] in b"\r\n":
                        raw_end -= 1
                    val = Stream(self, val, start, raw_end, num, closed)
                    streams.append((m.start(), val))
            prev = self.objs.get(num)
            if prev is None or prev[0] <= m.start():
                self.objs[num] = (m.start(), val)
        self.bad_objects = 0
        for at, s in streams:                          # objects inside object streams, each on its own
            if self.resolve(s.dict.get("Type")) != "ObjStm":
                continue
            try:
                body = s.decoded()[0]
                first, count = int(self.resolve(s.dict["First"])), int(self.resolve(s.dict["N"]))
                head = [t for t, _ in tokens(body[:first])][:2 * count]
            except Budget as err:
                raise Unreadable(str(err)) from None
            except Exception:  # noqa: BLE001 - an unreadable object stream leaves its objects out
                self.bad_objects += 1
                continue
            for k in range(0, len(head) - 1, 2):
                num, off = head[k], head[k + 1]
                if not isinstance(num, int) or not isinstance(off, int):
                    self.bad_objects += 1
                    continue
                prev = self.objs.get(num)
                if prev is not None and prev[0] >= at:
                    continue
                try:
                    self.objs[num] = (at, Parser(body, first + off).object())
                except (EOFError, ValueError, IndexError, RecursionError):
                    self.bad_objects += 1
        trailers = []
        for m in re.finditer(rb"trailer[\x00\t\n\x0c\r ]*<<", data):
            try:
                trailers.append((m.start(), Parser(data, m.end() - 2).object()))
            except (EOFError, ValueError, IndexError, RecursionError):
                continue
        trailers += [(at, v.dict) for at, v in self.objs.values() if isinstance(v, Stream) and v.dict.get("Type") == "XRef"]
        trailers.sort(key=lambda x: x[0])
        if any("Encrypt" in t for _, t in trailers if isinstance(t, dict)):
            raise Unreadable("encrypted")
        self.root = next((t["Root"] for _, t in reversed(trailers) if isinstance(t, dict) and "Root" in t), None)
        if self.resolve(self.root) is None:
            self.root = next((Ref((n, 0)) for n, (_, v) in sorted(self.objs.items())
                              if isinstance(v, dict) and v.get("Type") == "Catalog"), None)
        if not isinstance(self.resolve(self.root), dict):
            raise Unreadable("no document catalog")

    def resolve(self, x, depth=0):
        while isinstance(x, Ref) and depth < MAX_DEPTH:
            x = (self.objs.get(x[0]) or (None, None))[1]
            depth += 1
        return None if isinstance(x, Ref) else x

    def pages(self):
        out, seen = [], set()

        def walk(node, res, depth):
            key = node if isinstance(node, Ref) else None
            if key is not None:
                if key in seen:
                    return
                seen.add(key)
            d = self.resolve(node)
            if not isinstance(d, dict) or depth > 64:
                return
            res = d.get("Resources", res)
            kids = self.resolve(d.get("Kids"))
            if isinstance(kids, list):
                for k in kids:
                    walk(k, res, depth + 1)
            elif d.get("Type") == "Page" or "Contents" in d:
                out.append((d, res))
        walk(self.resolve(self.root).get("Pages"), None, 0)
        if not out:
            raise Unreadable("no page tree")
        return out


# ---------------------------------------------------------------- content streams

class Page:
    def __init__(self, doc):
        self.doc, self.parts, self.bad, self.fonts, self.problems, self.notes = doc, [], 0, {}, {}, []
        self.size, self.last_y, self.tokens, self.calls, self.chars = 10.0, None, 0, 0, 0
        self.scale, self.line_h = 1.0, 10.0     # the text matrix scale, and the height of the text last shown (layout only)

    def _resource(self, res, kind, name):
        r = self.doc.resolve(res)
        group = self.doc.resolve(r.get(kind)) if isinstance(r, dict) else None
        return group.get(str(name)) if isinstance(group, dict) else None, group

    def font(self, res, name):
        ref, fonts = self._resource(res, "Font", name)
        key = (str(name), ref if isinstance(ref, Ref) else id(fonts))
        if key not in self.fonts:
            self.fonts[key] = Font(self.doc, self.doc.resolve(ref), str(name)) if ref is not None else None
        return self.fonts[key]

    def show(self, font, s):
        if font is None:
            text, bad = REPLACEMENT * max(1, len(s)), max(1, len(s))
            self.problems["(no font)"] = "text shown without a font"
        else:
            text, bad = font.decode(s)
            if bad:
                self.problems[font.key + " " + font.basefont] = font.problem or "codes without a character"
        self.bad += bad
        self.line_h = self.size * self.scale
        self.chars += len(text)
        if self.chars > PAGE_CHARS:
            raise Budget("more than %d characters on the page" % PAGE_CHARS)
        self.parts.append(text)

    def newline(self):
        if self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    def space(self):
        if self.parts and not self.parts[-1].endswith((" ", "\n")):
            self.parts.append(" ")

    def content(self, stream):
        data, note = stream.decoded()
        if note and note not in self.notes:
            self.notes.append(note)
        return data

    def run(self, data, res, depth=0, seen=(), font=None):
        """Interpret one content stream; font is the caller's (a form XObject starts with it). The font and its size
        belong to the graphics state: q saves them, Q restores them, and the caller's state comes back afterwards."""
        size0 = self.size
        stack, ops, p = [], [], Parser(data)
        while True:
            try:
                t = p.next()
            except EOFError:
                break
            self.tokens += 1
            self.doc.tokens += 1
            if self.tokens > PAGE_TOKENS:
                raise Budget("more than %d tokens on the page" % PAGE_TOKENS)
            if self.doc.tokens > DOC_TOKENS:
                raise Budget("more than %d tokens in the document" % DOC_TOKENS)
            if not (isinstance(t, Op) and t not in ("[", "<<", "true", "false", "null")):
                try:
                    if len(ops) < MAX_OPERANDS:
                        ops.append(p.object(t))
                except EOFError:
                    break
                continue
            if t == "BI":                               # inline image: its data is skipped up to EI
                m = re.compile(rb"[\x00\t\n\x0c\r ]ID[\x00\t\n\x0c\r ]").search(data, p.pos - 1)
                e = m and re.compile(rb"[\x00\t\n\x0c\r ]EI(?=[\x00\t\n\x0c\r ]|$)").search(data, m.end())
                if not e:
                    break
                p.restart(e.end())
            elif t == "q":
                stack.append((font, self.size))
            elif t == "Q":
                if stack:
                    font, self.size = stack.pop()
            elif t == "Tf" and ops:
                font = self.font(res, ops[0])
                if len(ops) > 1 and isinstance(ops[1], (int, float)):
                    self.size = abs(ops[1]) or self.size
            elif t in ("Tj", "'", '"') and ops:
                if t != "Tj":
                    self.newline()
                if isinstance(ops[-1], bytes):
                    self.show(font, ops[-1])
            elif t == "TJ" and ops and isinstance(ops[-1], list):
                for x in ops[-1]:
                    if isinstance(x, bytes):
                        self.show(font, x)
                    elif isinstance(x, (int, float)) and x < -250:
                        self.space()
            elif t == "Do" and ops:
                self.form(ops[-1], res, depth, seen, font)
            else:
                self.position(t, ops)
            ops = []
        self.size = size0
        self.newline()

    def form(self, name, res, depth, seen, font):
        ref, _ = self._resource(res, "XObject", name)
        s = self.doc.resolve(ref)
        if not (isinstance(s, Stream) and s.dict.get("Subtype") == "Form") or ref in seen:
            return
        self.calls += 1
        if self.calls > PAGE_DO:
            raise Budget("more than %d form calls on the page" % PAGE_DO)
        if depth >= MAX_DEPTH:
            raise Budget("forms nested deeper than %d" % MAX_DEPTH)
        self.run(self.content(s), s.dict.get("Resources", res), depth + 1, tuple(seen) + (ref,), font)

    def position(self, t, ops):
        if t in ("Td", "TD") and len(ops) >= 2:
            tx, ty = ops[-2], ops[-1]
            if isinstance(ty, (int, float)) and abs(ty) > 0.01:
                self.newline()
            elif isinstance(tx, (int, float)) and tx > 0:
                self.space()
        elif t == "T*":
            self.newline()
        elif t == "Tm" and len(ops) >= 6 and isinstance(ops[-1], (int, float)):
            y, d = ops[-1], ops[-3] if isinstance(ops[-3], (int, float)) else 1
            self.scale = abs(d) or 1.0
            if self.last_y is not None and abs(y - self.last_y) > 0.5 * self.line_h:
                self.newline()
            else:
                self.space()
            self.last_y = y


# ---------------------------------------------------------------- readability

def _base(c):
    """A Latin letter without its accent, lower case ('É' -> 'e'); '' for a letter of another script."""
    if not unicodedata.name(c, "").startswith("LATIN"):
        return ""
    return unicodedata.normalize("NFD", c)[0].lower()


def readability(text):
    """The measures of a page's text: characters (whitespace aside), readable share, letter words of four letters or
    more (sequences aside) with the Latin share of their letters, and over their Latin letters (accents removed): the
    share of words with a vowel (a e i o u y, accented ones included), the vowel share (a e i o u) and the common-pair
    share. The Latin measures are None when the words are not mostly Latin."""
    chars = [c for c in text if not c.isspace()]
    bad = sum(1 for c in chars if c == REPLACEMENT or unicodedata.category(c) in ("Cc", "Co", "Cn", "Cs"))
    words = [w for w in re.findall(r"[^\W\d_]{4,}", text) if not SEQUENCE.fullmatch(w)]
    folded = ["".join(_base(c) for c in w) for w in words]
    letters = sum(len(w) for w in words)
    latin = sum(len(w) for w in folded)
    mostly_latin = bool(letters) and latin >= 0.5 * letters
    lw = [w for w in folded if len(w) >= 4]
    like = sum(1 for w in lw if re.search(r"[aeiouy]", w)) / len(lw) if mostly_latin and lw else None
    joined = "".join(lw)
    vowels = sum(1 for c in joined if c in "aeiou") / len(joined) if mostly_latin and joined else None
    pairs = [w[k:k + 2] for w in lw for k in range(len(w) - 1)]
    common = sum(1 for x in pairs if x in COMMON_PAIRS) / len(pairs) if mostly_latin and pairs else None
    return dict(characters=len(chars), readable=(len(chars) - bad) / len(chars) if chars else 0.0, words=len(words),
                latin=latin / letters if letters else None, latin_words=len(lw), wordlike=like, vowels=vowels, pairs=common)


def verdict(m, min_ratio=MIN_RATIO):
    """(status, reason) of a page from its measures: ok, unreadable, unverified (mostly non-Latin words, which the
    word measures cannot check) or no_text."""
    if m["characters"] == 0:
        return "no_text", "no text on the page (an image or a scan)"
    if m["readable"] < min_ratio:
        return "unreadable", "readable characters %.3f < %.2f" % (m["readable"], min_ratio)
    if m["words"] >= MIN_WORDS and m["latin"] is not None and m["latin"] < 0.5:
        return "unverified", "mostly non-Latin letters (%.3f Latin): the word measures cannot check the character map" % m["latin"]
    if m["latin_words"] >= MIN_WORDS:
        if m["wordlike"] < MIN_WORDLIKE:
            return "unreadable", "word-like words %.3f < %.2f (a character map that does not give words)" % (m["wordlike"], MIN_WORDLIKE)
        if not VOWELS[0] <= m["vowels"] <= VOWELS[1]:
            return "unreadable", "vowels %.3f of the letters, outside %.2f-%.2f (a shifted character map)" % (m["vowels"], *VOWELS)
    if m["latin_words"] >= MIN_WORDS_PAIRS and m["pairs"] < MIN_PAIRS:
        return "unreadable", "common letter pairs %.3f < %.2f (a shifted character map)" % (m["pairs"], MIN_PAIRS)
    return "ok", None


def _clean(text):
    text = "".join(c for c in text if unicodedata.category(c) != "Cf" or c == "­")
    text = text.replace(" ", " ").replace("­", "-")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip("\n")


def _escape_text(text):
    """Page text as written: control and separator characters escaped, lines starting with '#' or '\\' given a '\\'."""
    lines = safe(text, keep="\n\t").split("\n")
    return "\n".join("\\" + x if x.startswith(("#", "\\")) else x for x in lines)


def extract(data, min_ratio=MIN_RATIO):
    """The readable text of a PDF given as bytes, page by page, with the reason for every page left out. Every string
    from the file in the result has its control characters escaped."""
    out = dict(pdf_text=VERSION, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), min_ratio=min_ratio, pages=[])
    try:
        doc = Document(data)
        pages = doc.pages()
    except Unreadable as err:
        out.update(status="unreadable", reason=safe(err))
        return out
    except Exception as err:  # noqa: BLE001 - any structural failure makes the whole file unreadable
        out.update(status="unreadable", reason="structure not read (%s)" % type(err).__name__)
        return out
    fonts = {}
    for k, (d, res) in enumerate(pages, 1):
        pg = Page(doc)
        rec = dict(page=k)
        try:
            contents = doc.resolve(d.get("Contents"))
            parts = contents if isinstance(contents, list) else [contents]
            body = b"\n".join(pg.content(s) for s in (doc.resolve(x) for x in parts) if isinstance(s, Stream))
            pg.run(body, res)
        except (Budget, ValueError) as err:
            rec.update(status="unreadable", reason=safe(err))
            out["pages"].append(rec)
            continue
        except Exception as err:  # noqa: BLE001
            rec.update(status="unreadable", reason="content not read (%s)" % type(err).__name__)
            out["pages"].append(rec)
            continue
        for f in pg.fonts.values():
            if f is not None:
                fonts[safe("%s %s" % (f.key, f.basefont))] = safe(f.how or f.problem)
        text = _clean("".join(pg.parts))
        m = readability(text)
        rec.update({key: (round(v, 4) if isinstance(v, float) else v) for key, v in m.items()})
        status, reason = verdict(m, min_ratio)
        problems = "; ".join(safe("%s: %s" % kv) for kv in sorted(pg.problems.items()))
        notes = "; ".join([safe(x) for x in pg.notes] + ([problems] if problems else []))
        rec["status"] = status
        if status == "ok":
            rec["text"] = _escape_text(text)
            if notes:
                rec["note"] = notes
        else:
            rec["reason"] = reason + ("; " + notes if notes else "")
        out["pages"].append(rec)
    ok = sum(1 for p in out["pages"] if p["status"] == "ok")
    out["fonts"] = dict(sorted(fonts.items()))
    out["bad_objects"] = doc.bad_objects
    out["status"] = "ok" if ok == len(out["pages"]) else ("partial" if ok else "unreadable")
    if not ok:
        out["reason"] = "no readable page"
    return out


def render(name, doc):
    """The text file: a header, then per page a '## page N' line with its status and, for readable pages only, the text."""
    head = ("# %s sha256 %s; pdf_text %s; status %s; %d pages, %d readable; thresholds: characters %.2f, word-like %.2f, "
            "vowels %.2f-%.2f, common pairs %.2f" % (
                safe(name), doc["sha256"], doc["pdf_text"], doc["status"], len(doc["pages"]),
                sum(1 for p in doc["pages"] if p["status"] == "ok"), doc["min_ratio"], MIN_WORDLIKE, VOWELS[0], VOWELS[1], MIN_PAIRS))
    lines = [head + ("; " + doc["reason"] if doc.get("reason") else "")]
    for p in doc["pages"]:
        if p["status"] == "ok":
            lines.append("## page %d: ok (%d characters, %.3f readable)%s" % (
                p["page"], p["characters"], p["readable"], "; note: " + p["note"] if p.get("note") else ""))
            lines.append(p["text"])
        else:
            lines.append("## page %d: %s (%s)" % (p["page"], p["status"], p["reason"]))
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pdf")
    ap.add_argument("--sha256", help="the checksum the file was stored with; any other file is refused")
    a = ap.parse_args(argv)
    with open(a.pdf, "rb") as f:
        data = f.read()
    if a.sha256 and hashlib.sha256(data).hexdigest() != a.sha256.lower():
        sys.stderr.write("pdf_text: %s is not the stored file (sha256 differs)\n" % safe(a.pdf))
        return 2
    sys.stdout.write(render(a.pdf.rsplit("/", 1)[-1], extract(data)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""bench/rel/registry/pdf_text.py: readable PDFs give their text, unreadable ones give none, garbage never passes, and
no file can make the extractor run long or write a false line."""
from __future__ import annotations

import base64
import hashlib
import random
import time
import zlib

import pytest

from bench.rel.registry import pdf_text as P


def _pdf(objects, trailer=b""):
    out = b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n"
    for i, body in enumerate(objects, 1):
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    return out + b"trailer\n<< /Size %d /Root 1 0 R %s>>\nstartxref\n0\n%%%%EOF\n" % (len(objects) + 1, trailer)


def _stream(content, flate=True, extra=b""):
    data = zlib.compress(content) if flate else content
    return (b"<< /Length %d %s%s>>\nstream\n" % (len(data), b"/Filter /FlateDecode " if flate else b"", extra)
            + data + b"\nendstream")


CMAP = b"""/CIDInit /ProcSet findresource begin 12 dict begin begincmap
/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def
1 begincodespacerange <0000> <FFFF> endcodespacerange
2 beginbfchar <0003> <0020> <0010> <00E9> endbfchar
1 beginbfrange <0024> <003D> <0041> endbfrange
endcmap CMapName currentdict /CMap defineresource pop end end"""


def _codes(text):
    """The two-byte codes of CMAP for capitals, space and e-acute."""
    out = b""
    for c in text:
        out += (3 if c == " " else 0x10 if c == "é" else 0x24 + ord(c) - 65).to_bytes(2, "big")
    return b"<" + out.hex().encode() + b">"


def _pages(*pages):
    """objects: catalog, page tree, then per page (page, font, contents, extra stream or null); the font is /F1."""
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", None]
    kids = []
    for font, content, extra in pages:
        n = len(objs) + 1
        kids.append(b"%d 0 R" % n)
        objs.append(b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>" % (n + 1, n + 2))
        objs.append(font.replace(b"@TU", b"%d 0 R" % (n + 3)))
        objs.append(_stream(content))
        objs.append(_stream(extra) if extra is not None else b"null")
    objs[1] = b"<< /Type /Pages /Kids [" + b" ".join(kids) + b"] /Count %d >>" % len(kids)
    return _pdf(objs)


HELV = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
TYPE0 = b"<< /Type /Font /Subtype /Type0 /BaseFont /ABCDEF+Arial /Encoding /Identity-H /DescendantFonts [] /ToUnicode @TU >>"
TYPE0_BARE = b"<< /Type /Font /Subtype /Type0 /BaseFont /ABCDEF+Arial /Encoding /Identity-H /DescendantFonts [] >>"
LINE1 = b"BT /F1 12 Tf 72 700 Td (Dataset S2: LGT fragments in Alloteropsis) Tj 0 -14 Td [(gene) -300 (ASEM\\137AUS1\\13712345)] TJ ET"
PROSE = ("Lateral gene transfer moves genetic material between organisms that are not parent and offspring. In grasses, several "
         "studies have reported nuclear genes that appear to have been acquired from distantly related species. The evidence "
         "usually combines a phylogeny in which the recipient gene sits among donor sequences with strong support, the position "
         "of the gene in the recipient genome, and the absence of the gene from close relatives of the recipient. Each of these "
         "lines of evidence can be wrong on its own: contamination can place foreign reads in an assembly, long branches can "
         "attract each other in a tree, and gene loss can mimic a patchy distribution. A registry of transfer events therefore "
         "records which evidence was reported for every claim, and how strongly each claim is supported.")


def _lines(text, width=80):
    lines, cur = [], ""
    for w in text.split():
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + [cur]


def _content(text, font=b"/F1"):
    body = b" 0 -12 Td ".join(b"(" + x.encode("latin-1") + b") Tj" for x in _lines(text))
    return b"BT " + font + b" 10 Tf 72 700 Td " + body + b" ET"


def _shifted_font(n):
    """A simple font whose Differences name every letter n places on: a valid-looking but wrong character map."""
    diffs = b" ".join(b"%d " % base + b" ".join(b"/" + bytes([(k + n) % 26 + base]) for k in range(26)) for base in (65, 97))
    return (b"<< /Type /Font /Subtype /Type1 /BaseFont /ABCDEF+Body /Encoding << /BaseEncoding /WinAnsiEncoding "
            b"/Differences [" + diffs + b"] >> >>")


# ---------------------------------------------------------------- readable and unreadable

def test_a_readable_pdf_gives_its_text_page_by_page():
    data = _pages((HELV, LINE1, None), (TYPE0, b"BT /F1 10 Tf 72 700 Td " + _codes("LGT CAFé") + b" Tj ET", CMAP))
    got = P.extract(data)
    assert got["status"] == "ok" and got["sha256"] == hashlib.sha256(data).hexdigest()
    assert got["pages"][0]["text"] == "Dataset S2: LGT fragments in Alloteropsis\ngene ASEM_AUS1_12345"
    assert got["pages"][1]["text"] == "LGT CAFé" and got["pages"][1]["readable"] == 1.0
    assert got["fonts"] == {"F1 Helvetica": "encoding", "F1 Arial": "ToUnicode"}
    out = P.render("s2.pdf", got)
    assert out.startswith("# s2.pdf sha256 %s; pdf_text %s; status ok; 2 pages, 2 readable" % (got["sha256"], P.VERSION))
    assert "## page 2: ok (7 characters, 1.000 readable)\nLGT CAFé\n" in out


def test_real_english_reads_and_every_letter_shift_of_it_is_refused():
    assert P.extract(_pages((_shifted_font(0), _content(PROSE), None)))["status"] == "ok"
    for n in range(1, 26):
        page = P.extract(_pages((_shifted_font(n), _content(PROSE), None)))["pages"][0]
        assert page["status"] == "unreadable" and "text" not in page, n


def test_a_shifted_tounicode_map_is_refused():
    """A composite font whose ToUnicode is off by one: every code is mapped, the characters are wrong."""
    caps = "".join(c for c in PROSE.upper() if "A" <= c <= "Z" or c == " ")
    shifted = CMAP.replace(b"1 beginbfrange <0024> <003D> <0041> endbfrange",
                           b"2 beginbfrange <0024> <003C> <0042> <003D> <003D> <0041> endbfrange")
    assert P.extract(_pages((TYPE0, b"BT /F1 10 Tf " + _codes(caps) + b" Tj ET", CMAP)))["status"] == "ok"
    page = P.extract(_pages((TYPE0, b"BT /F1 10 Tf " + _codes(caps) + b" Tj ET", shifted)))["pages"][0]
    assert page["status"] == "unreadable" and "text" not in page


def test_an_unreadable_font_gives_no_text():
    data = _pages((TYPE0_BARE, b"BT /F1 10 Tf 72 700 Td " + _codes("LGT CAFé") + b" Tj ET", None))
    got = P.extract(data)
    page = got["pages"][0]
    assert got["status"] == "unreadable" and page["status"] == "unreadable" and "text" not in page
    assert "composite font without ToUnicode (encoding /Identity-H" in page["reason"] and page["readable"] == 0.0
    assert "LGT" not in P.render("x.pdf", got)


def test_garbage_never_passes_the_thresholds():
    pua = CMAP.replace(b"<0024> <003D> <0041>", b"<0024> <003D> <E000>")             # private-use characters
    consonants = CMAP.replace(b"<0024> <003D> <0041>", b"<0024> <003D> <0042>").replace(b"<0010> <00E9>", b"<0010> <0042>")
    words = " ".join(["ABCF", "GJKL", "MPQR", "SVWB"] * 6)       # shown through the shifted map: BCDG HKLM NQRS TWXC, no word
    good = (HELV, LINE1, None)
    for bad in ((TYPE0, b"BT /F1 10 Tf 72 700 Td " + _codes("LGT CASE") + b" Tj ET", pua),
                (TYPE0, b"BT /F1 10 Tf 72 700 Td " + _codes(words) + b" Tj ET", consonants)):
        got = P.extract(_pages(good, bad))
        assert got["status"] == "partial" and got["pages"][0]["status"] == "ok"
        assert got["pages"][1]["status"] == "unreadable" and "text" not in got["pages"][1]
        out = P.render("x.pdf", got)
        assert "## page 2: unreadable (" in out and out.count("\n") == 5
    reason = P.extract(_pages(good, (TYPE0, b"BT /F1 10 Tf 72 700 Td " + _codes(words) + b" Tj ET", consonants)))["pages"][1]["reason"]
    assert reason.startswith("word-like words 0.000 < 0.90")
    assert "readable characters 0.000 < 0.95" in P.extract(_pages(good, (TYPE0, b"BT /F1 10 Tf " + _codes("LGT") + b" Tj ET", pua)))["pages"][1]["reason"]


def test_threshold_boundaries():
    def page(text):
        return P.extract(_pages((HELV, b"BT /F1 12 Tf (" + text + b") Tj ET", None)))["pages"][0]
    assert page(b"abcdefghijklmnopqrs" + b"\\201")["status"] == "ok"                     # 19 readable and 1 U+FFFD: 0.95
    assert page(b"abcdefghijklmnopqr" + b"\\201" * 2)["status"] == "unreadable"         # 18 and 2: 0.90
    assert page(b" ".join([b"gene"] * 18 + [b"bcdf"] * 2))["status"] == "ok"              # word-like 18 of 20: 0.90
    assert page(b" ".join([b"gene"] * 17 + [b"bcdf"] * 3))["reason"].startswith("word-like words 0.850 < 0.90")
    assert page(b"abcde" * 19 + b"\\201" * 5)["status"] == "ok"                       # 95 of 100 readable: 0.95
    assert page(b"abcde" * 19 + b"\\201" * 6)["status"] == "unreadable"               # 95 of 101
    assert page(b" ".join([b"bcdf"] * 19))["status"] == "ok"                           # 19 words: the word measures wait
    assert page(b" ".join([b"bcdf"] * 20))["reason"].startswith("word-like words 0.000 < 0.90")
    assert page(b" ".join([b"aqoz"] * 19))["status"] == "ok"
    assert page(b" ".join([b"aqoz"] * 20))["reason"].startswith("common letter pairs 0.000 < 0.45")
    assert page(b" ".join([b"abcd"] * 20))["reason"].startswith("vowels 0.250 of the letters, outside 0.30-0.55")
    assert page(b" ".join([b"aeio"] * 20))["reason"].startswith("vowels 1.000")
    assert page(b" ".join([b"ACGTTGCAAGCT"] * 30))["status"] == "ok"                  # sequences are not words
    m = P.readability(PROSE)
    assert m["wordlike"] == 1.0 and 0.30 <= m["vowels"] <= 0.55 and m["pairs"] > 0.70


def test_a_rare_unmapped_code_stays_visible_and_noted():
    content = b"BT /F1 12 Tf 72 700 Td (" + b"Readable text of a supplement " * 4 + b"\\201) Tj ET"   # 0x81: no WinAnsi character
    page = P.extract(_pages((HELV, content, None)))["pages"][0]
    assert page["status"] == "ok" and page["text"].endswith(P.REPLACEMENT) and "codes without a character" in page["note"]


# ---------------------------------------------------------------- fonts and encodings

def test_glyph_names_and_differences_without_a_base_encoding():
    assert P.glyph("fi") == "fi" and P.glyph("f_i") == "fi" and P.glyph("uni00660069") == "fi"
    assert P.glyph("eacute") == "é" and P.glyph("Alpha") == "Α" and P.glyph("quoteright") == "’"
    assert P.glyph("A.sc") == "A" and P.glyph("u1F600") == "\U0001F600"
    assert P.glyph("g77") is None and P.glyph("a1") is None and P.glyph("uniD800") is None
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /ABCDEF+CMR10 /Encoding << /Differences [65 /A /B /fi /quoteright /g77] >> >>"
    got = P.extract(_pages((font, b"BT /F1 12 Tf (" + b"ABCD" * 10 + b"EF) Tj ET", None)))
    assert got["fonts"] == {"F1 CMR10": "Differences only"}
    assert got["pages"][0]["text"].startswith("ABfi’ABfi’") and got["pages"][0]["text"].endswith(P.REPLACEMENT * 2)
    plain = b"<< /Type /Font /Subtype /Type1 /BaseFont /ABCDEF+CMR10 >>"                # its own encoding is not stated
    got = P.extract(_pages((plain, b"BT /F1 12 Tf (Some words) Tj ET", None)))
    assert got["status"] == "unreadable" and "the font program's own encoding is not read" in got["pages"][0]["reason"]
    std = b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>"                   # a standard 14 font: Standard encoding
    assert P.extract(_pages((std, b"BT /F1 12 Tf (It's here) Tj ET", None)))["pages"][0]["text"] == "It’s here"
    subset = b"<< /Type /Font /Subtype /Type1 /BaseFont /ABCDEF+Times-Roman >>"         # an embedded subset: its own encoding
    assert P.extract(_pages((subset, b"BT /F1 12 Tf (It's here) Tj ET", None)))["status"] == "unreadable"


def test_macroman_as_the_pdf_standard_defines_it():
    font = b"<< /Type /Font /Subtype /TrueType /BaseFont /Geneva /Encoding /MacRomanEncoding >>"
    page = P.extract(_pages((font, b"BT /F1 12 Tf (caf\\216 na\\225ve and more plain words here x \\255 y) Tj ET", None)))["pages"][0]
    assert page["text"] == "café naïve and more plain words here x %s y" % P.REPLACEMENT   # 0xAD (not equal): not in it
    enc = P._encoding("MacRomanEncoding")
    assert not set(enc) & P.MACROMAN_UNDEFINED and len(P.MACROMAN_UNDEFINED) == 15 and enc[0xDB] == "¤"


def test_bfranges_stay_ranges():
    space, single, ranges = P._parse_cmap(b"1 begincodespacerange <0000> <FFFF> endcodespacerange "
                                          b"1 beginbfrange <0000> <FFFF> <0000> endbfrange "
                                          b"1 beginbfrange <0100> <0102> [<0041> <0042> <0043>] endbfrange")
    assert single == {} and len(ranges) == 2
    font = P.Font.__new__(P.Font)
    font.space, font.single, font.ranges, font.byte, font.memo, font.unicode_cmap = space, single, ranges, {}, {}, False
    assert font.decode(b"\x00\x41\x01\x01\xff\xfe") == ("Aā￾", 0)          # the first range wins, as listed


def test_no_object_repr_in_a_reason():
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
            b"<< /Type /Font /Subtype /Type0 /BaseFont /X /Encoding 6 0 R >>", _stream(b"BT /F1 10 Tf <0001> Tj ET"),
            _stream(b"/CMapName /Custom def")]
    reason = P.extract(_pdf(objs))["pages"][0]["reason"]
    assert "an embedded stream (object 6)" in reason and " at 0x" not in reason and "object at" not in reason


# ---------------------------------------------------------------- graphics state

def test_q_and_q_restore_the_font_and_forms_start_with_the_callers():
    other = b"<< /Type /Font /Subtype /Type1 /BaseFont /ABCDEF+Body /Encoding << /Differences [65 /Z] >> >>"
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R /F2 5 0 R >> /XObject << /Fm1 7 0 R /Fm2 8 0 R >> >>"
            b" /Contents 6 0 R >>", HELV, other,
            _stream(b"BT /F1 12 Tf (Above) Tj ET q BT /F2 12 Tf 0 -14 Td (A) Tj ET Q BT 0 -14 Td (Below) Tj ET "
                    b"/Fm1 Do BT 0 -14 Td (After) Tj ET /Fm2 Do BT 0 -14 Td (Last) Tj ET"),
            _stream(b"BT 0 -14 Td (Inherited) Tj ET", extra=b"/Type /XObject /Subtype /Form "),
            _stream(b"BT /F2 12 Tf 0 -14 Td (A) Tj ET", extra=b"/Type /XObject /Subtype /Form ")]
    assert P.extract(_pdf(objs))["pages"][0]["text"] == "Above\nZ\nBelow\nInherited\nAfter\nZ\nLast"


# ---------------------------------------------------------------- limits

def test_a_stream_or_a_document_over_its_limit_is_unreadable(monkeypatch):
    monkeypatch.setattr(P, "STREAM_MAX", 1 << 20)
    big = b"BT /F1 12 Tf (x) Tj ET " + b" " * (2 << 20)
    page = P.extract(_pages((HELV, big, None)))["pages"][0]
    assert page == dict(page=1, status="unreadable", reason="a stream decodes to more than 1 MB")
    monkeypatch.setattr(P, "STREAM_MAX", 64 << 20)
    monkeypatch.setattr(P, "DOC_MAX", 3 << 20)
    half = b"BT /F1 12 Tf (x) Tj ET " + b" " * (2 << 20)
    got = P.extract(_pages((HELV, half, None), (HELV, half + b" ", None)))
    assert got["pages"][0]["status"] == "ok" and got["pages"][1]["status"] == "unreadable"
    assert got["pages"][1]["reason"] == "a stream decodes to more than 1 MB"           # what is left of the document's 3 MB


def test_lzw_output_is_limited_and_its_table_stops_at_4096():
    codes, size, width = [], 258, 9                  # KwKwK: every code the next new entry, then the longest one again
    for k in range(5000):
        code = 65 if k == 0 else (size if size < 4096 else 4095)
        codes.append((code, width))
        if k and size < 4096:
            size += 1
        if k and size + 1 >= (1 << width) and width < 12:
            width += 1
    bits = format(256, "09b") + "".join(format(c, "0%db" % w) for c, w in codes)
    bits += "0" * (-len(bits) % 8)
    data = bytes(int(bits[k:k + 8], 2) for k in range(0, len(bits), 8))
    with pytest.raises(P.Budget):
        P._lzw(data, 1, 1 << 20)
    out, note = P._lzw(data, 1, 64 << 20)
    full = 4096 - 258                                 # codes that add an entry after the first
    assert len(out) == 1 + sum(range(2, full + 2)) + (5000 - 1 - full) * (full + 1) and note == "LZW data without an end code"


def test_a_broken_flate_stream_is_read_up_to_the_break_quickly():
    rnd = random.Random(1)
    vocab = ["".join(rnd.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rnd.randint(3, 9))) for _ in range(3000)]
    text = " ".join(rnd.choice(vocab) for _ in range(600_000)).encode()
    comp = bytearray(zlib.compress(text))
    mid = len(comp) // 2
    comp[mid:mid + 8] = b"\xff" * 8
    t0 = time.time()
    with pytest.raises(P.Broken) as err:                 # zlib noticed only at the checksum: what came before was wrong
        P._inflate(bytes(comp), 64 << 20)
    assert time.time() - t0 < 5 and str(err.value).startswith("compressed data broken at byte ")
    assert int(str(err.value).split("byte ")[1].split(" ")[0]) >= mid - P.CHUNK
    early = bytearray(zlib.compress(text))
    early[200:240] = bytes(range(40))                     # broken near the start: found byte by byte in the first piece
    t0 = time.time()
    with pytest.raises(P.Broken) as err:
        P._inflate(bytes(early), 64 << 20)
    assert time.time() - t0 < 5 and int(str(err.value).split("byte ")[1].split(" ")[0]) < P.CHUNK
    cut, note = P._inflate(zlib.compress(b"a whole stream " * 100)[:-20], 1 << 20)
    assert note.startswith("compressed data cut short") and (b"a whole stream " * 100).startswith(cut) and cut
    whole = zlib.compress(b"BT /F1 12 Tf (Kept) Tj ET")
    page = P.extract(_pages((HELV, b"BT /F1 12 Tf (Kept) Tj ET", None)).replace(whole, whole[:-4]))["pages"][0]
    assert page["status"] == "ok" and page["text"] == "Kept" and "cut short" in page["note"]    # no checksum: cut short
    page = P.extract(_pages((HELV, b"BT /F1 12 Tf (Kept) Tj ET", None)).replace(whole, whole[:-4] + b"\x00\x00\x00\x00"))["pages"][0]
    assert page["status"] == "unreadable" and "text" not in page and "compressed data broken at byte" in page["reason"]


def test_form_fan_out_is_bounded_and_streams_are_decoded_once(monkeypatch):
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /Resources << /XObject << /X 5 0 R >> /Font << /F1 4 0 R >> >> /Contents 9 0 R >>", HELV]
    for level in range(4):                               # objects 5..8: each calls the next 20 times
        objs.append(_stream(b"/X Do " * 20, extra=b"/Type /XObject /Subtype /Form /Resources << /XObject << /X %d 0 R >> >> " % (6 + level)))
    objs[-1] = _stream(b"BT /F1 12 Tf (leaf) Tj ET", extra=b"/Type /XObject /Subtype /Form /Resources << /Font << /F1 4 0 R >> >> ")
    objs.append(_stream(b"/X Do " * 20))
    calls = []
    real = P.decode
    monkeypatch.setattr(P, "decode", lambda *a, **k: calls.append(1) or real(*a, **k))
    t0 = time.time()
    page = P.extract(_pdf(objs))["pages"][0]
    assert time.time() - t0 < 5 and page["status"] == "unreadable" and page["reason"].startswith("more than 2000 form calls")
    assert len(calls) == 5                                # the page's content and four forms, each decoded once
    monkeypatch.setattr(P, "PAGE_TOKENS", 1000)
    assert P.extract(_pages((HELV, b"1 " * 2000 + b"BT /F1 12 Tf (x) Tj ET", None)))["pages"][0]["reason"] == "more than 1000 tokens on the page"


# ---------------------------------------------------------------- escaping

def test_names_from_the_file_cannot_forge_a_line():
    font = TYPE0_BARE.replace(b"/BaseFont /ABCDEF+Arial", b"/BaseFont /X#0A##20page#202:#20ok")
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>",
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F#0A##20page#201:#20ok 4 0 R >> >> /Contents 5 0 R >>",
            font, _stream(b"BT /F#0A##20page#201:#20ok 10 Tf <0001> Tj ET"),
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 7 0 R >> >> /Contents 8 0 R >>", HELV,
            _stream(b"BT /F1 12 Tf (# not a header) Tj 0 -14 Td (\\\\ a backslash line) Tj ET")]
    got = P.extract(_pdf(objs))
    out = P.render("x.pdf", got)
    heads = [x for x in out.split("\n") if x.startswith("#")]
    assert len(heads) == 3 and heads[1].startswith("## page 1: unreadable (") and heads[2].startswith("## page 2: ok (")
    assert "\\x0a" in got["pages"][0]["reason"] and "\n" not in got["pages"][0]["reason"]
    assert got["pages"][1]["text"] == "\\# not a header\n\\\\ a backslash line"
    assert all("\n" not in k and "\n" not in v for k, v in got["fonts"].items())
    bell = CMAP.replace(b"2 beginbfchar", b"3 beginbfchar <0004> <0007>")              # a map that gives a control character
    shown = _codes("BELL") + b" Tj <0004> Tj " + _codes("HERE AND MANY MORE READABLE LETTERS")
    page = P.extract(_pages((TYPE0, b"BT /F1 10 Tf " + shown + b" Tj ET", bell)))["pages"][0]
    assert page["text"] == "BELL\\x07HERE AND MANY MORE READABLE LETTERS"


# ---------------------------------------------------------------- structure and filters

def test_objects_inside_an_object_stream_one_broken_object_aside():
    body = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    head = b"9 0 4 2 "                                   # object 9 at offset 0 is broken, object 4 at offset 2 is the font
    objstm = _stream(head + b"] " + body, extra=b"/Type /ObjStm /N 2 /First %d " % len(head))
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>", b"null",
            _stream(b"BT /F1 12 Tf (From an object stream) Tj ET"), objstm]
    data = _pdf(objs).replace(b"4 0 obj\nnull\nendobj\n", b"")
    got = P.extract(data)
    assert got["pages"][0]["text"] == "From an object stream"


def test_encrypted_and_broken_files_are_unreadable():
    data = _pages((HELV, LINE1, None))
    enc = data.replace(b"/Root 1 0 R", b"/Root 1 0 R /Encrypt << /Filter /Standard >>")
    assert P.extract(enc) == dict(P.extract(enc), status="unreadable", reason="encrypted", pages=[])
    assert P.extract(b"not a pdf at all")["reason"] == "not a PDF (no %PDF- header)"
    assert P.extract(b"%PDF-1.4\n1 0 obj << /Type /Font >> endobj\n")["reason"] == "no document catalog"


def test_a_page_without_text_and_a_form_xobject_with_text():
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>",
            b"<< /Type /Page /Parent 2 0 R /Resources << /XObject << /Im1 8 0 R >> >> /Contents 4 0 R >>",
            _stream(b"q 100 0 0 100 0 0 cm /Im1 Do Q"), HELV,
            b"<< /Type /Page /Parent 2 0 R /Resources << /XObject << /Fm1 7 0 R >> >> /Contents 9 0 R >>",
            _stream(b"BT /F1 12 Tf (Text inside a form) Tj ET", extra=b"/Type /XObject /Subtype /Form /Resources << /Font << /F1 5 0 R >> >> "),
            _stream(b"\x00\xff" * 10, flate=False, extra=b"/Type /XObject /Subtype /Image /Filter /DCTDecode "),
            _stream(b"/Fm1 Do")]
    got = P.extract(_pdf(objs))
    assert got["pages"][0]["status"] == "no_text" and got["pages"][1]["text"] == "Text inside a form"
    assert got["status"] == "partial"


def _lzw_encode(data):
    table, w, codes = {bytes([i]): i for i in range(256)}, b"", [256]
    for c in data:
        wc = w + bytes([c])
        if wc in table:
            w = wc
            continue
        codes.append(table[w])
        table[wc] = len(table) + 2
        w = bytes([c])
    codes += [table[w], 257]
    bits = "".join(format(c, "09b") for c in codes)
    bits += "0" * (-len(bits) % 8)
    return bytes(int(bits[k:k + 8], 2) for k in range(0, len(bits), 8))


def _runlength_encode(data):
    out = b""
    for k in range(0, len(data), 100):
        chunk = data[k:k + 100]
        out += bytes([len(chunk) - 1]) + chunk
    return out + b"\x80"


def test_the_filters_decode():
    text = b"TOBEORNOTTOBEORTOBEORNOT and more text of a supplement"
    assert P.decode("LZWDecode", _lzw_encode(text)) == (text, None)
    assert P.decode("ASCII85Decode", base64.a85encode(text, adobe=True)) == (text, None)
    assert P.decode("ASCIIHexDecode", text.hex().encode() + b">") == (text, None)
    assert P.decode("RunLengthDecode", _runlength_encode(text)) == (text, None) and P.decode("RunLengthDecode", b"\xfdA\x80")[0] == b"AAAA"
    chained = _stream(base64.a85encode(zlib.compress(b"BT /F1 12 Tf (Chained filters) Tj ET"), adobe=True), flate=False,
                      extra=b"/Filter [/ASCII85Decode /FlateDecode] /DecodeParms [null] ")       # a DecodeParms array shorter than the filters
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>", HELV, chained]
    assert P.extract(_pdf(objs))["pages"][0]["text"] == "Chained filters"
    jbig = _pages((HELV, LINE1, None)).replace(b"/Filter /FlateDecode", b"/Filter /JBIG2Decode", 1)
    assert P.extract(jbig)["pages"][0] == dict(page=1, status="unreadable", reason="filter /JBIG2Decode is not read")
    two = _pages((HELV, LINE1, None)).replace(b"/Filter /FlateDecode", b"/Filter [/FlateDecode /ASCIIHexDecode] /DecodeParms << >>", 1)
    assert "DecodeParms does not match the 2 filters" in P.extract(two)["pages"][0]["reason"]


def test_predictors():
    rows = bytes([2, 1, 2, 3, 2, 1, 1, 1])               # PNG Up rows of 3 bytes: 1 2 3 then 2 3 4
    assert P._predict(rows, dict(Predictor=12, Columns=3)) == bytes([1, 2, 3, 2, 3, 4])
    assert P._predict(bytes([1, 1, 1]), dict(Predictor=2, Columns=3)) == bytes([1, 2, 3])
    for bad in (dict(Predictor=3), dict(Predictor=2, Columns=3, BitsPerComponent=4), dict(Predictor=12, Columns=0)):
        with pytest.raises(ValueError):
            P._predict(b"\x00\x01\x02\x03", bad)
    with pytest.raises(ValueError):
        P._predict(bytes([7, 1, 2, 3]), dict(Predictor=12, Columns=3))


def test_inline_images_are_skipped():
    content = b"BT /F1 12 Tf (Before) Tj ET BI /W 2 /H 2 /BPC 8 /CS /G ID \x00(Tj)\xff) Tj EI BT /F1 12 Tf 0 -20 Td (after) Tj ET"
    assert P.extract(_pages((HELV, content, None)))["pages"][0]["text"] == "Before\nafter"


def test_the_command_refuses_a_file_that_is_not_the_stored_one(tmp_path, capsys):
    p = tmp_path / "s2.pdf"
    data = _pages((HELV, LINE1, None))
    p.write_bytes(data)
    assert P.main([str(p), "--sha256", "0" * 64]) == 2
    assert P.main([str(p), "--sha256", hashlib.sha256(data).hexdigest().upper()]) == 0
    assert capsys.readouterr().out.startswith("# s2.pdf sha256 " + hashlib.sha256(data).hexdigest())


def test_the_reviewers_q_and_q_example():
    """q/Q keep the font: F1 again after Q, so the third piece is 'def', not 'xyz' through F2."""
    f2 = b"<< /Type /Font /Subtype /Type1 /BaseFont /ABCDEF+Body /Encoding << /Differences [100 /x /y /z] >> >>"
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>", HELV, f2,
            _stream(b"BT /F1 Tf (abc) Tj ET q BT /F2 Tf (d) Tj ET Q BT (def) Tj ET")]
    page = P.extract(_pdf(objs))["pages"][0]
    assert page["status"] == "ok" and page["text"] == "abc" + "x" + "def"
    objs[5] = _stream(b"Q Q BT /F1 Tf (abc) Tj ET Q")                                     # Q on an empty stack: ignored
    assert P.extract(_pdf(objs))["pages"][0]["text"] == "abc"


def test_this_modules_own_english_reads():
    m = P.readability(P.__doc__)
    assert P.verdict(m) == ("ok", None) and m["latin_words"] > 100


def test_cid_codes_read_as_unicode_and_shifted_are_refused():
    """A composite font whose ToUnicode passes the codes through, the codes being the characters plus one: every code
    is 'mapped', the text is shifted (spaces become '!')."""
    identity = (b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap 1 begincodespacerange <0000> <FFFF> "
                b"endcodespacerange 1 beginbfrange <0000> <FFFF> <0000> endbfrange endcmap end end")
    shown = b"<" + "".join("%04x" % (ord(c) + 1) for c in PROSE).encode() + b">"
    page = P.extract(_pages((TYPE0, b"BT /F1 10 Tf " + shown + b" Tj ET", identity)))["pages"][0]
    assert page["status"] == "unreadable" and "text" not in page
    plain = b"<" + "".join("%04x" % ord(c) for c in PROSE).encode() + b">"
    assert P.extract(_pages((TYPE0, b"BT /F1 10 Tf " + plain + b" Tj ET", identity)))["pages"][0]["status"] == "ok"


def test_random_letter_substitutions_are_refused():
    rnd = random.Random(7)
    letters = "abcdefghijklmnopqrstuvwxyz"
    passed = 0
    for _ in range(2000):
        perm = list(letters)
        rnd.shuffle(perm)
        table = str.maketrans(letters + letters.upper(), "".join(perm) + "".join(perm).upper())
        passed += P.verdict(P.readability(PROSE.translate(table)))[0] == "ok"
    assert passed == 0


def test_a_page_of_another_script_is_unverified_and_accents_count_only_on_vowels():
    greek = CMAP.replace(b"<0024> <003D> <0041>", b"<0024> <003D> <0391>")             # capitals through to Greek capitals
    words = " ".join(["ABCDE", "FGHIK", "LMNOP", "ABCDQ"] * 6)        # A to Q: Greek capitals with no gap (U+03A2 is none)
    page = P.extract(_pages((TYPE0, b"BT /F1 10 Tf " + _codes(words) + b" Tj ET", greek)))["pages"][0]
    assert page["status"] == "unverified" and "text" not in page and page["reason"].startswith("mostly non-Latin letters")
    m = P.readability(" ".join(["\u0141\u0143\u0158\u0160"] * 20))                    # Latin letters, accented consonants only
    assert m["wordlike"] == 0.0 and P.verdict(m)[0] == "unreadable"
    m = P.readability(" ".join(["\u00e9t\u00e9s", "caf\u00e9s", "na\u00efve", "\u00e9cole", "th\u00e9\u00e2tre"] * 4))
    assert m["wordlike"] == 1.0 and m["latin"] == 1.0

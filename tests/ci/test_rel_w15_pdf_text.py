"""bench/rel/registry/pdf_text.py: readable PDFs give their text, unreadable ones give none, garbage never passes."""
from __future__ import annotations

import base64
import hashlib
import zlib

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
    """objects: catalog, page tree, then per page (page, font, contents[, extra])."""
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", None]
    kids = []
    for font, content, extra in pages:
        n = len(objs) + 1
        kids.append(b"%d 0 R" % n)
        objs.append(b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>" % (n + 1, n + 2))
        objs.append(font.replace(b"@TU", b"%d 0 R" % (n + 3)))
        objs.append(_stream(content))
        if extra is not None:
            objs.append(_stream(extra))
        else:
            objs.append(b"null")
    objs[1] = b"<< /Type /Pages /Kids [" + b" ".join(kids) + b"] /Count %d >>" % len(kids)
    return _pdf(objs)


HELV = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
TYPE0 = b"<< /Type /Font /Subtype /Type0 /BaseFont /ABCDEF+Arial /Encoding /Identity-H /DescendantFonts [] /ToUnicode @TU >>"
TYPE0_BARE = b"<< /Type /Font /Subtype /Type0 /BaseFont /ABCDEF+Arial /Encoding /Identity-H /DescendantFonts [] >>"
LINE1 = b"BT /F1 12 Tf 72 700 Td (Dataset S2: LGT fragments in Alloteropsis) Tj 0 -14 Td [(gene) -300 (ASEM\\137AUS1\\13712345)] TJ ET"


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


def test_an_unreadable_font_gives_no_text():
    data = _pages((TYPE0_BARE, b"BT /F1 10 Tf 72 700 Td " + _codes("LGT CAFé") + b" Tj ET", None))
    got = P.extract(data)
    page = got["pages"][0]
    assert got["status"] == "unreadable" and page["status"] == "unreadable" and "text" not in page
    assert "composite font without ToUnicode" in page["reason"] and page["readable"] == 0.0
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
        assert "## page 2: unreadable (" in out and out.count("\n") == 5         # header, page 1 line and its 2 lines, page 2
    reason = P.extract(_pages(good, (TYPE0, b"BT /F1 10 Tf 72 700 Td " + _codes(words) + b" Tj ET", consonants)))["pages"][1]["reason"]
    assert reason.startswith("word-like words 0.000 < 0.75")
    assert "readable characters 0.000 < 0.95" in P.extract(_pages(good, (TYPE0, b"BT /F1 10 Tf " + _codes("LGT") + b" Tj ET", pua)))["pages"][1]["reason"]


def test_a_rare_unmapped_code_stays_visible_and_noted():
    content = b"BT /F1 12 Tf 72 700 Td (" + b"Readable text of a supplement " * 4 + b"\\201) Tj ET"   # 0x81: no WinAnsi character
    got = P.extract(_pages((HELV, content, None)))
    page = got["pages"][0]
    assert page["status"] == "ok" and page["text"].endswith(P.REPLACEMENT) and "codes without a character" in page["note"]


def test_glyph_names_and_differences_without_a_base_encoding():
    assert P.glyph("fi") == "fi" and P.glyph("f_i") == "fi" and P.glyph("uni00660069") == "fi"
    assert P.glyph("eacute") == "é" and P.glyph("Alpha") == "Α" and P.glyph("quoteright") == "’"
    assert P.glyph("A.sc") == "A" and P.glyph("u1F600") == "\U0001F600"
    assert P.glyph("g77") is None and P.glyph("a1") is None and P.glyph("uniD800") is None
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /ABCDEF+CMR10 /Encoding << /Differences [65 /A /B /fi /quoteright /g77] >> >>"
    got = P.extract(_pages((font, b"BT /F1 12 Tf (" + b"ABCD" * 30 + b"EF) Tj ET", None)))
    assert got["fonts"] == {"F1 CMR10": "Differences only"}
    assert got["pages"][0]["text"].startswith("ABfi’ABfi’") and got["pages"][0]["text"].endswith(P.REPLACEMENT * 2)
    plain = b"<< /Type /Font /Subtype /Type1 /BaseFont /ABCDEF+CMR10 >>"                # its own encoding is not stated
    got = P.extract(_pages((plain, b"BT /F1 12 Tf (Some words) Tj ET", None)))
    assert got["status"] == "unreadable" and "the font program's own encoding is not read" in got["pages"][0]["reason"]
    std = b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>"                   # a standard 14 font: Standard encoding
    assert P.extract(_pages((std, b"BT /F1 12 Tf (It's here) Tj ET", None)))["pages"][0]["text"] == "It’s here"


def test_objects_inside_an_object_stream():
    body = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    head = b"4 0 "
    objstm = _stream(head + body, extra=b"/Type /ObjStm /N 1 /First %d " % len(head))
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>", b"null",
            _stream(b"BT /F1 12 Tf (From an object stream) Tj ET"), objstm]
    data = _pdf(objs).replace(b"4 0 obj\nnull\nendobj\n", b"")
    assert P.extract(data)["pages"][0]["text"] == "From an object stream"


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
    assert P.decode("LZWDecode", _lzw_encode(text)) == text
    assert P.decode("ASCII85Decode", base64.a85encode(text, adobe=True)) == text
    assert P.decode("ASCIIHexDecode", text.hex().encode() + b">") == text
    assert P.decode("RunLengthDecode", _runlength_encode(text)) == text + b"" and P.decode("RunLengthDecode", b"\xfdA\x80") == b"AAAA"
    assert P.decode("FlateDecode", zlib.compress(text)[:-6]).startswith(b"TOBEORNOT")          # cut short: what decompresses
    chained = _stream(base64.a85encode(zlib.compress(b"BT /F1 12 Tf (Chained filters) Tj ET"), adobe=True), flate=False,
                      extra=b"/Filter [/ASCII85Decode /FlateDecode] ")
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>", HELV, chained]
    assert P.extract(_pdf(objs))["pages"][0]["text"] == "Chained filters"
    dct = _pages((HELV, LINE1, None)).replace(b"/Filter /FlateDecode", b"/Filter /JBIG2Decode", 1)
    assert P.extract(dct)["pages"][0] == dict(page=1, status="unreadable", reason="filter JBIG2Decode is not read")


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

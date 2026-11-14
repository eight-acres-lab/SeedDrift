"""Serve web/ on localhost (the page then talks to the SeedDrift API, e.g. through an SSH tunnel)."""
from __future__ import annotations

import functools
import http.server
import os
import webbrowser

from .config import HOME


def serve(port=8700, api="http://localhost:8600"):
    web = os.path.join(HOME, "web")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=web)
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        url = "http://localhost:%d/" % port
        print("SeedDrift web page: %s   (API: %s; change it at the top of the page)" % (url, api))
        try:
            webbrowser.open(url)
        except Exception:                      # noqa: BLE001 - headless machines
            pass
        httpd.serve_forever()

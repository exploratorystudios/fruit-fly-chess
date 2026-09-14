import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastchess.serverless import respond  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        respond(self, 'candidates')
